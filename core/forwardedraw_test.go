package bifrost

import (
	"context"
	"encoding/json"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/providers/anthropic"
	"github.com/maximhq/bifrost/core/providers/anthropic/anthropictest"
	"github.com/maximhq/bifrost/core/schemas"
)

// converterTarget is one (provider, model) whose request conversion is one branch of core's
// dispatch.
type converterTarget struct {
	provider schemas.ModelProvider
	model    string
}

// converterTargets are a representative model per converter branch of every provider whose chat
// and Responses conversion core runs against a plain base URL.
var converterTargets = []converterTarget{
	{schemas.OpenAI, "gpt-4o"},
	{schemas.Azure, "gpt-4o"},
	{schemas.Azure, "claude-sonnet-4-5"},
	{schemas.Vertex, "gemini-2.5-flash"},
	{schemas.Vertex, "meta/llama-3.3-70b-instruct-maas"},
	{schemas.Vertex, "claude-sonnet-4-5"},
	{schemas.Gemini, "gemini-2.5-flash"},
	{schemas.Bedrock, "amazon.nova-pro-v1:0"},
	{schemas.Cohere, "command-r-plus"},
	{schemas.Replicate, "meta/meta-llama-3-70b-instruct"},
	{schemas.Perplexity, "sonar"},
	{schemas.Perplexity, "openai/gpt-5"},
	{schemas.Anthropic, "claude-sonnet-4-5"},
	{schemas.Cerebras, "llama-3.3-70b"},
	{schemas.DeepSeek, "deepseek-chat"},
	{schemas.Groq, "llama-3.3-70b-versatile"},
	{schemas.Mistral, "mistral-large-latest"},
	{schemas.OpenRouter, "openai/gpt-4o"},
	{schemas.XAI, "grok-4"},
	{schemas.Fireworks, "accounts/fireworks/models/llama-v3p3-70b-instruct"},
	{schemas.Nebius, "meta-llama/Llama-3.3-70B-Instruct"},
	{schemas.Parasail, "parasail-llama-33-70b-fp8"},
	{schemas.Sarvam, "sarvam-m"},
	{schemas.MiniMax, "MiniMax-M2"},
	{schemas.BytePlus, "seed-1-6"},
	{schemas.Baseten, "deepseek-ai/DeepSeek-V3"},
	{schemas.Kimchi, "kimchi-1"},
	{schemas.Runware, "runware-1"},
	{schemas.Wafer, "wafer-1"},
	{schemas.GithubCopilot, "gpt-4o"},
	{schemas.OpencodeGo, "go-1"},
	{schemas.OpencodeZen, "zen-1"},
	{schemas.Ollama, "llama3"},
	{schemas.SGL, "llama3"},
	{schemas.VLLM, "llama3"},
	{schemas.HuggingFace, "nebius/meta-llama/Llama-3.3-70B-Instruct"},
}

// recordingUpstream records every request body and header set, answering a 400 the provider
// passes back: only the request matters.
type recordingUpstream struct {
	mu      sync.Mutex
	srv     *httptest.Server
	bodies  []string
	headers []string
}

func newRecordingUpstream(t *testing.T, tls bool) *recordingUpstream {
	t.Helper()
	u := &recordingUpstream{}
	h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/token" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"access_token":"t","token_type":"Bearer","expires_in":3600}`)
			return
		}
		b, _ := io.ReadAll(r.Body)
		var hs strings.Builder
		for k, v := range r.Header {
			hs.WriteString(k + ":" + strings.Join(v, ",") + "\n")
		}
		u.mu.Lock()
		u.bodies = append(u.bodies, string(b))
		u.headers = append(u.headers, hs.String())
		u.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":{"message":"recorded","type":"invalid_request_error"}}`)
	})
	if tls {
		u.srv = httptest.NewTLSServer(h)
	} else {
		u.srv = httptest.NewServer(h)
	}
	t.Cleanup(u.srv.Close)
	return u
}

// take returns and clears what was recorded.
func (u *recordingUpstream) take() (body, headers string) {
	u.mu.Lock()
	defer u.mu.Unlock()
	body, headers = strings.Join(u.bodies, "\n"), strings.Join(u.headers, "\n")
	u.bodies, u.headers = nil, nil
	return body, headers
}

func converterClient(t *testing.T, provider schemas.ModelProvider, u *recordingUpstream) *Bifrost {
	t.Helper()
	url := u.srv.URL
	account := NewMockAccount()
	account.AddProviderWithBaseURL(provider, 1, 4, url)
	key := schemas.Key{ID: "k", Value: *schemas.NewSecretVar("sk-k"), Models: schemas.WhiteList{"*"}, Weight: 100}
	switch provider {
	case schemas.Azure:
		key.AzureKeyConfig = &schemas.AzureKeyConfig{Endpoint: *schemas.NewSecretVar(url)}
	case schemas.Vertex:
		creds := `{"type":"authorized_user","client_id":"c","client_secret":"s","refresh_token":"r","token_uri":"` + url + `/token"}`
		key.VertexKeyConfig = &schemas.VertexKeyConfig{ProjectID: *schemas.NewSecretVar("p"), Region: *schemas.NewSecretVar("us-central1"),
			AuthCredentials: *schemas.NewSecretVar(creds)}
	case schemas.Bedrock:
		key.BedrockKeyConfig = &schemas.BedrockKeyConfig{Region: schemas.NewSecretVar("us-east-1"),
			Endpoints: &schemas.BedrockEndpoints{Runtime: schemas.NewSecretVar(strings.TrimPrefix(url, "https://"))}}
		account.configs[provider].NetworkConfig.InsecureSkipVerify = true
	case schemas.VLLM:
		key.VLLMKeyConfig = &schemas.VLLMKeyConfig{URL: *schemas.NewSecretVar(url)}
	case schemas.Ollama:
		key.OllamaKeyConfig = &schemas.OllamaKeyConfig{URL: *schemas.NewSecretVar(url)}
	case schemas.SGL:
		key.SGLKeyConfig = &schemas.SGLKeyConfig{URL: *schemas.NewSecretVar(url)}
	}
	nc := &account.configs[provider].NetworkConfig
	nc.MaxRetries, nc.AllowPrivateNetwork = 0, true
	nc.RetryBackoffInitial, nc.RetryBackoffMax = time.Millisecond, time.Millisecond
	account.SetKeysForProvider(provider, []schemas.Key{key})
	client, err := Init(context.Background(), schemas.BifrostConfig{Account: account, Logger: NewDefaultLogger(schemas.LogLevelError)})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(client.Shutdown)
	return client
}

// forwardTokens is, per feature, the token its member leaves in an upstream body (or header)
// that forwards it.
var forwardTokens = map[string]string{
	"container": "cprobe1", "container_skills": "skillprobe", "mcp_servers": "mcpprobe", "speed": `"speed"`,
	"task_budget": "task_budget", "inference_geo": "inference_geo", "diagnostics": "cache_diagnosis",
	"safeguards": "safeguards", "cache_control_scope": `"scope"`, "fallback_credit_token": "tokprobe",
	"context_management": "compact_20260112", "defer_loading": "defer_loading", "allowed_callers": "allowed_callers",
	"input_examples": "input_examples", "eager_input_streaming": "eager_input_streaming", "server_tool": "web_fetch",
	"tool_version": "web_search_20260209", "anthropic_beta": anthropictest.MCPClientBeta, "fallback_block": "fallback",
	"compaction_block": "compaction", "tool_change_block": "tool_addition",
}

// forwardResponses is how a Responses request carries row: its Responses shape, else its raw key.
func forwardResponses(row anthropictest.FeatureRow) func(*schemas.BifrostResponsesRequest) {
	if row.Responses != nil || len(row.Raw) == 0 {
		return row.Responses
	}
	return func(r *schemas.BifrostResponsesRequest) {
		r.Params.ExtraParams = make(map[string]any, len(row.Raw))
		for k, v := range row.Raw {
			r.Params.ExtraParams[k] = json.RawMessage(v)
		}
	}
}

func converterRequestCtx(passthrough bool, betas []string) *schemas.BifrostContext {
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	if passthrough {
		ctx.SetValue(schemas.BifrostContextKeyPassthroughExtraParams, true)
	}
	if len(betas) > 0 {
		ctx.SetValue(schemas.BifrostContextKeyClientAnthropicBeta, betas)
	}
	return ctx
}

func baseChatRequest(tg converterTarget) *schemas.BifrostChatRequest {
	return &schemas.BifrostChatRequest{Provider: tg.provider, Model: tg.model,
		Input:  []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("hi")}}},
		Params: &schemas.ChatParameters{MaxCompletionTokens: schemas.Ptr(64)}}
}

func baseResponsesRequest(tg converterTarget) *schemas.BifrostResponsesRequest {
	return &schemas.BifrostResponsesRequest{Provider: tg.provider, Model: tg.model,
		Input:  []schemas.ResponsesMessage{{Role: schemas.Ptr(schemas.ResponsesInputMessageRoleUser), Content: &schemas.ResponsesMessageContent{ContentStr: schemas.Ptr("hi")}}},
		Params: &schemas.ResponsesParameters{MaxOutputTokens: schemas.Ptr(64)}}
}

// send runs one request through the client and returns what reached the upstream.
func send(t *testing.T, client *Bifrost, u *recordingUpstream, ctx *schemas.BifrostContext, chat *schemas.BifrostChatRequest, resp *schemas.BifrostResponsesRequest) (string, string) {
	t.Helper()
	if chat != nil {
		_, _ = client.ChatCompletionRequest(ctx, chat)
	} else {
		_, _ = client.ResponsesRequest(ctx, resp)
	}
	body, headers := u.take()
	if body == "" {
		t.Fatalf("no request reached the upstream")
	}
	return body, headers
}

// AC-5b: for every converter branch, kind and passthrough setting, a request carrying only one
// feature member puts it in the upstream request if and only if ForwardedRaw says so. Claude
// branches are excluded: their builders carry members, which is Uncarried's question.
func TestForwardedRaw_AgreesWithConverters(t *testing.T) {
	rows := anthropictest.FeatureRows()
	for _, tg := range converterTargets {
		if schemas.IsAnthropicModelFamily(nil, tg.model) {
			continue
		}
		u := newRecordingUpstream(t, tg.provider == schemas.Bedrock)
		client := converterClient(t, tg.provider, u)
		for _, kind := range []schemas.RequestType{schemas.ChatCompletionRequest, schemas.ResponsesRequest} {
			for _, passthrough := range []bool{false, true} {
				name := string(tg.provider) + "/" + tg.model + "/" + string(kind)
				if passthrough {
					name += "/passthrough"
				}
				t.Run(name, func(t *testing.T) {
					forwarded := anthropic.ForwardedRaw(tg.provider, tg.model, kind, passthrough)
					baseChat, baseResp := baseChatRequest(tg), baseResponsesRequest(tg)
					if kind == schemas.ChatCompletionRequest {
						baseResp = nil
					} else {
						baseChat = nil
					}
					baseBody, baseHeaders := send(t, client, u, converterRequestCtx(passthrough, nil), baseChat, baseResp)
					var observed anthropic.Feature
					for _, row := range rows {
						chat, resp := baseChatRequest(tg), baseResponsesRequest(tg)
						if kind == schemas.ChatCompletionRequest {
							resp = nil
							if row.Chat != nil {
								row.Chat(chat)
							} else if row.Betas == nil {
								continue
							}
						} else {
							chat = nil
							if respond := forwardResponses(row); respond != nil {
								respond(resp)
							} else if row.Betas == nil {
								continue
							}
						}
						token := forwardTokens[row.Name]
						if token == "" {
							t.Fatalf("no forward token for %s", row.Name)
						}
						if strings.Contains(baseBody+baseHeaders, token) {
							t.Fatalf("%s: the baseline already holds %q", row.Name, token)
						}
						body, headers := send(t, client, u, converterRequestCtx(passthrough, row.Betas), chat, resp)
						if strings.Contains(body+headers, token) {
							observed |= row.Feature()
						}
					}
					if observed != forwarded {
						t.Errorf("ForwardedRaw = %q, the converter forwards %q", forwarded.String(), observed.String())
					}
				})
			}
		}
	}
}

// extraKeyValues are plausible values for every ExtraParams key some converter reads, so a
// converter that reads one changes its upstream body.
var extraKeyValues = map[string]string{
	"top_k":                             `5`,
	"cached_content":                    `"cachedContents/probe"`,
	"cachedContent":                     `"cachedContents/probe"`,
	"safety_settings":                   `[{"category":"HARM_CATEGORY_HATE_SPEECH","threshold":"BLOCK_NONE"}]`,
	"safetySettings":                    `[{"category":"HARM_CATEGORY_HATE_SPEECH","threshold":"BLOCK_NONE"}]`,
	"labels":                            `{"probe":"label"}`,
	"response_mime_type":                `"application/json"`,
	"response_schema":                   `{"type":"object","properties":{"probe":{"type":"string"}}}`,
	"response_json_schema":              `{"type":"object","properties":{"probe":{"type":"string"}}}`,
	"response_modalities":               `["TEXT"]`,
	"response_logprobs":                 `true`,
	"media_resolution":                  `"MEDIA_RESOLUTION_LOW"`,
	"candidate_count":                   `2`,
	"seed":                              `7`,
	"stop_sequences":                    `["probestop"]`,
	"additionalModelRequestFieldPaths":  `{"probe":1}`,
	"additionalModelResponseFieldPaths": `["/probe"]`,
	"guardrailConfig":                   `{"guardrailIdentifier":"g","guardrailVersion":"1"}`,
	"performanceConfig":                 `{"latency":"optimized"}`,
	"requestMetadata":                   `{"probe":"meta"}`,
	"promptVariables":                   `{"probe":{"text":"v"}}`,
	"context_management":                `{"edits":[{"type":"clear_tool_uses_20250919"}]}`,
	"safeguards":                        `{"mode":"auto"}`,
	"reasoning_summary":                 `"detailed"`,
	"thinking":                          `{"type":"enabled","budget_tokens":1024}`,
	"inference_geo":                     `"us"`,
	"speed":                             `"fast"`,
	"diagnostics":                       `{"cache_diagnosis":true}`,
	"fallbacks":                         `[{"model":"claude-opus-4-7"}]`,
	"fallback_credit_token":             `"tokprobe"`,
	"safety_mode":                       `"STRICT"`,
	"log_probs":                         `true`,
	"strict_tool_choice":                `true`,
	"prompt_cache_key":                  `"probekey"`,
	"container":                         `"cprobe1"`,
	"task_budget":                       `{"type":"tokens","total":4242}`,
	"x_unknown_probe":                   `{"a":1}`,
}

// AC-5b: with passthrough off, a request carrying one ExtraParams key reaches the upstream
// changed if and only if ConsumedExtraKeys names the key, for every converter branch (Claude
// included) and kind. Every key some converter reads is tried on every branch, plus one unknown.
func TestConsumedExtraKeys_AgreesWithConverters(t *testing.T) {
	for _, tg := range converterTargets {
		u := newRecordingUpstream(t, tg.provider == schemas.Bedrock)
		client := converterClient(t, tg.provider, u)
		for _, kind := range []schemas.RequestType{schemas.ChatCompletionRequest, schemas.ResponsesRequest} {
			t.Run(string(tg.provider)+"/"+tg.model+"/"+string(kind), func(t *testing.T) {
				build := func(key string) (*schemas.BifrostChatRequest, *schemas.BifrostResponsesRequest) {
					var extra map[string]any
					if key != "" {
						var v any
						if err := json.Unmarshal([]byte(extraKeyValues[key]), &v); err != nil {
							t.Fatal(err)
						}
						extra = map[string]any{key: v}
					}
					if kind == schemas.ChatCompletionRequest {
						r := baseChatRequest(tg)
						r.Params.ExtraParams = extra
						return r, nil
					}
					r := baseResponsesRequest(tg)
					r.Params.ExtraParams = extra
					return nil, r
				}
				chat, resp := build("")
				baseline, _ := send(t, client, u, converterRequestCtx(false, nil), chat, resp)
				table := anthropic.ConsumedExtraKeys(tg.provider, tg.model, kind)
				all := anthropic.ConsumesAllExtras(tg.provider, tg.model, kind)
				var got []string
				for key := range extraKeyValues {
					chat, resp := build(key)
					ctx := converterRequestCtx(false, nil)
					body, _ := send(t, client, u, ctx, chat, resp)
					// A key the converter reads changes the body, or is removed by a strip that
					// records it; an unread key is dropped silently.
					if body != baseline || anthropic.RemovedFeatures(ctx) != 0 || schemas.AdaptationsFromContext(ctx) != 0 {
						got = append(got, key)
					}
				}
				slices.Sort(got)
				want := slices.Clone(table)
				if all {
					want = slices.Sorted(maps.Keys(extraKeyValues))
				}
				slices.Sort(want)
				if !slices.Equal(got, want) {
					t.Errorf("ConsumedExtraKeys = %q, the converter reads %q", want, got)
				}
			})
		}
	}
}

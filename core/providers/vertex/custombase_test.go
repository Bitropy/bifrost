package vertex

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
	"golang.org/x/oauth2"
)

// customVertexKey is the custom provider every golden case below runs next to native vertex.
const customVertexKey = schemas.ModelProvider("vertex-eu")

// vertexWireRequest is what the upstream saw: everything a provider-family decision can change.
type vertexWireRequest struct {
	Method, Path, Query string
	Header              http.Header
	Body                string
}

const (
	goldenGeminiJSON = `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],` +
		`"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2},"modelVersion":"m"}`
	goldenGeminiSpeechJSON = `{"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"audio/L16;rate=24000","data":"AAAA"}}]},"finishReason":"STOP"}],` +
		`"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`
	goldenClaudeJSON = `{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5","content":[{"type":"text","text":"ok"}],` +
		`"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`
	goldenClaudeSSE = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-5\",\"content\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
)

// vertexWireLog collects the requests the recorder saw.
type vertexWireLog struct {
	mu   sync.Mutex
	reqs []vertexWireRequest
}

// take returns the requests seen since the last take.
func (l *vertexWireLog) take() []vertexWireRequest {
	l.mu.Lock()
	defer l.mu.Unlock()
	reqs := l.reqs
	l.reqs = nil
	return reqs
}

// newVertexWireRecorder answers every Vertex method the golden cases call and logs each request.
func newVertexWireRecorder(t *testing.T) (*httptest.Server, *vertexWireLog) {
	t.Helper()
	seen := &vertexWireLog{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		header := r.Header.Clone()
		header.Del("Content-Length") // follows the body, which is compared itself
		seen.mu.Lock()
		seen.reqs = append(seen.reqs, vertexWireRequest{r.Method, r.URL.Path, r.URL.RawQuery, header, string(body)})
		seen.mu.Unlock()
		path := r.URL.Path
		switch {
		case strings.HasSuffix(path, ":streamRawPredict"):
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, goldenClaudeSSE)
		case strings.HasSuffix(path, ":rawPredict"):
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, goldenClaudeJSON)
		case strings.HasSuffix(path, ":streamGenerateContent"):
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "data: "+goldenGeminiJSON+"\n\n")
		case strings.Contains(path, "tts"):
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, goldenGeminiSpeechJSON)
		default:
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, goldenGeminiJSON)
		}
	}))
	t.Cleanup(srv.Close)
	return srv, seen
}

// newGoldenVertex builds a Vertex provider on baseURL, native when custom is nil. Both carry the
// same ExtraHeaders beta list, so the Anthropic beta filter has something to drop.
func newGoldenVertex(t *testing.T, baseURL string, custom *schemas.CustomProviderConfig) (*VertexProvider, schemas.Key) {
	t.Helper()
	config := &schemas.ProviderConfig{
		NetworkConfig: schemas.NetworkConfig{
			BaseURL: baseURL, AllowPrivateNetwork: true, DefaultRequestTimeoutInSeconds: 5,
			ExtraHeaders: map[string]schemas.SecretVar{"Anthropic-Beta": *schemas.NewSecretVar("context-1m-2025-08-07,files-api-2025-04-14,totally-unknown-2099")},
		},
		CustomProviderConfig: custom,
	}
	config.CheckAndSetDefaults()
	provider, err := NewVertexProvider(config, vertexEndpointTestLogger{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(provider.client.CloseIdleConnections)
	key := schemas.Key{VertexKeyConfig: &schemas.VertexKeyConfig{
		ProjectID: *schemas.NewSecretVar("proj-eu"), Region: *schemas.NewSecretVar("europe-west4"), AuthCredentials: *schemas.NewSecretVar("test-credentials"),
	}}
	provider.tokenSources.Store(getClientKey(vertexCredentialIdentity(key)), oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "t"}))
	return provider, key
}

// goldenContext is the context the worker hands a Vertex-based attempt: the base provider type is
// vertex for the native provider and for a custom one alike.
func goldenContext(raw bool) *schemas.BifrostContext {
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	ctx.SetValue(schemas.BifrostContextKeyBaseProviderType, schemas.Vertex)
	ctx.SetValue(schemas.BifrostContextKeyIsCustomProvider, false)
	if raw {
		ctx.SetValue(schemas.BifrostContextKeyUseRawRequestBody, true)
	}
	return ctx
}

func goldenRequest[T any](t *testing.T, provider schemas.ModelProvider, body string) *T {
	t.Helper()
	var req T
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatalf("request fixture: %v", err)
	}
	reflect.ValueOf(&req).Elem().FieldByName("Provider").Set(reflect.ValueOf(provider))
	return &req
}

func goldenNoHooks(_ *schemas.BifrostContext, resp *schemas.BifrostResponse, err *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError) {
	return resp, err
}

// drainGoldenStream returns a reader that drains a stream to its end.
func drainGoldenStream(t *testing.T) func(chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	return func(stream chan *schemas.BifrostStreamChunk, bifrostErr *schemas.BifrostError) {
		t.Helper()
		if bifrostErr != nil {
			t.Fatalf("stream: %s", bifrostErr.GetErrorString())
		}
		timeout := time.After(5 * time.Second)
		for {
			select {
			case chunk, ok := <-stream:
				if !ok {
					return
				}
				if chunk.BifrostError != nil {
					t.Fatalf("stream error: %s", chunk.BifrostError.GetErrorString())
				}
			case <-timeout:
				t.Fatal("stream did not end")
			}
		}
	}
}

func checkGolden(t *testing.T, err *schemas.BifrostError) {
	t.Helper()
	if err != nil {
		t.Fatalf("request: %s", err.GetErrorString())
	}
}

// Request fixtures. Each carries the members a Vertex family decision rewrites: thinking budgets,
// mixed tool kinds, structured output, Anthropic server tools whose version Vertex downgrades,
// and a final assistant turn.
const (
	goldenGeminiChat = `{"model":"gemini-2.5-flash","input":[{"role":"system","content":"be brief"},{"role":"user","content":"hi"}],` +
		`"params":{"max_completion_tokens":64,"temperature":0.2,"reasoning":{"max_tokens":1024},"service_tier":"priority",` +
		`"tools":[{"type":"function","function":{"name":"get_weather","description":"d","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}}`
	goldenGeminiResponses = `{"model":"gemini-3-pro-preview","input":[{"type":"message","role":"user","content":"hi"},{"type":"message","role":"assistant","content":"partial"}],` +
		`"params":{"max_output_tokens":64,"reasoning":{"effort":"high"},"tools":[{"type":"function","name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}},{"type":"web_search"}]}}`
	goldenClaudeChat = `{"model":"claude-sonnet-4-5","input":[{"role":"system","content":"be brief"},{"role":"user","content":"hi"}],` +
		`"params":{"max_completion_tokens":4096,"reasoning":{"max_tokens":1024},` +
		`"response_format":{"type":"json_schema","json_schema":{"name":"out","schema":{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}}},` +
		`"tools":[{"type":"function","function":{"name":"get_weather","description":"d","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]}}`
	goldenClaudeChatNoLimit = `{"model":"claude-sonnet-4-5","input":[{"role":"user","content":"hi"}]}`
	goldenClaudeResponses   = `{"model":"claude-sonnet-4-5","input":[{"type":"message","role":"user","content":"hi"}],` +
		`"params":{"max_output_tokens":4096,"reasoning":{"effort":"high"},` +
		`"text":{"format":{"type":"json_schema","name":"out","schema":{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}}},` +
		`"tools":[{"type":"function","name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}},{"type":"web_search"}]}}`
	goldenGeminiRawBody   = `{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"thinkingConfig":{"thinkingBudget":1024},"presencePenalty":0.5},"fallbacks":["x"]}`
	goldenClaudeRouteBody = `{"model":"claude-sonnet-4-5","max_tokens":64,"tools":[{"type":"web_search_20260209","name":"web_search","max_uses":2}],` +
		`"messages":[{"role":"user","content":"hi"}]}`
	goldenSpeech = `{"model":"gemini-2.5-flash-preview-tts","input":{"input":"hello"},"params":{"voice":"Kore","response_format":"wav"}}`
)

type goldenCase struct {
	name string
	// path names the project and region the key routes to.
	path string
	call func(t *testing.T, p *VertexProvider, key schemas.Key)
}

// goldenBodyHas lists fragments a case's native body must carry for the case to test anything.
var goldenBodyHas = map[string]string{"claude chat, datasheet max tokens": `"max_tokens":1234`}

func goldenCases() []goldenCase {
	const geminiPath = "/v1/projects/proj-eu/locations/europe-west4/publishers/google/models/"
	const claudePath = "/v1/projects/proj-eu/locations/europe-west4/publishers/anthropic/models/"
	const claudeBetaPath = "/v1beta1/projects/proj-eu/locations/europe-west4/publishers/anthropic/models/"
	passthrough := func(stream bool, route schemas.ClaudeRoute, path, body string) func(t *testing.T, p *VertexProvider, key schemas.Key) {
		return func(t *testing.T, p *VertexProvider, key schemas.Key) {
			req := &schemas.BifrostPassthroughRequest{
				Method: http.MethodPost, Path: path, Model: "claude-sonnet-4-5", Body: []byte(body), ClaudeRoute: route,
				SafeHeaders: map[string]string{"Content-Type": "application/json", "anthropic-beta": "interleaved-thinking-2025-05-14,files-api-2025-04-14"},
			}
			if stream {
				drainGoldenStream(t)(p.PassthroughStream(goldenContext(false), goldenNoHooks, nil, key, req))
				return
			}
			_, err := p.Passthrough(goldenContext(false), key, req)
			checkGolden(t, err)
		}
	}
	return []goldenCase{
		{"gemini chat", geminiPath + "gemini-2.5-flash:generateContent", func(t *testing.T, p *VertexProvider, key schemas.Key) {
			_, err := p.ChatCompletion(goldenContext(false), key, goldenRequest[schemas.BifrostChatRequest](t, p.GetProviderKey(), goldenGeminiChat))
			checkGolden(t, err)
		}},
		{"gemini chat stream", geminiPath + "gemini-2.5-flash:streamGenerateContent", func(t *testing.T, p *VertexProvider, key schemas.Key) {
			drainGoldenStream(t)(p.ChatCompletionStream(goldenContext(false), goldenNoHooks, nil, key, goldenRequest[schemas.BifrostChatRequest](t, p.GetProviderKey(), goldenGeminiChat)))
		}},
		{"gemini chat raw body", geminiPath + "gemini-3-pro-preview:generateContent", func(t *testing.T, p *VertexProvider, key schemas.Key) {
			req := &schemas.BifrostChatRequest{Provider: p.GetProviderKey(), Model: "gemini-3-pro-preview", RawRequestBody: []byte(goldenGeminiRawBody)}
			_, err := p.ChatCompletion(goldenContext(true), key, req)
			checkGolden(t, err)
		}},
		{"gemini chat raw body stream", geminiPath + "gemini-3-pro-preview:streamGenerateContent", func(t *testing.T, p *VertexProvider, key schemas.Key) {
			req := &schemas.BifrostChatRequest{Provider: p.GetProviderKey(), Model: "gemini-3-pro-preview", RawRequestBody: []byte(goldenGeminiRawBody)}
			drainGoldenStream(t)(p.ChatCompletionStream(goldenContext(true), goldenNoHooks, nil, key, req))
		}},
		{"gemini responses", geminiPath + "gemini-3-pro-preview:generateContent", func(t *testing.T, p *VertexProvider, key schemas.Key) {
			_, err := p.Responses(goldenContext(false), key, goldenRequest[schemas.BifrostResponsesRequest](t, p.GetProviderKey(), goldenGeminiResponses))
			checkGolden(t, err)
		}},
		{"gemini responses stream", geminiPath + "gemini-3-pro-preview:streamGenerateContent", func(t *testing.T, p *VertexProvider, key schemas.Key) {
			drainGoldenStream(t)(p.ResponsesStream(goldenContext(false), goldenNoHooks, nil, key, goldenRequest[schemas.BifrostResponsesRequest](t, p.GetProviderKey(), goldenGeminiResponses)))
		}},
		{"gemini speech", geminiPath + "gemini-2.5-flash-preview-tts:generateContent", func(t *testing.T, p *VertexProvider, key schemas.Key) {
			_, err := p.Speech(goldenContext(false), key, goldenRequest[schemas.BifrostSpeechRequest](t, p.GetProviderKey(), goldenSpeech))
			checkGolden(t, err)
		}},
		{"claude chat", claudePath + "claude-sonnet-4-5:rawPredict", func(t *testing.T, p *VertexProvider, key schemas.Key) {
			_, err := p.ChatCompletion(goldenContext(false), key, goldenRequest[schemas.BifrostChatRequest](t, p.GetProviderKey(), goldenClaudeChat))
			checkGolden(t, err)
		}},
		{"claude chat, datasheet max tokens", claudePath + "claude-sonnet-4-5:rawPredict", func(t *testing.T, p *VertexProvider, key schemas.Key) {
			_, err := p.ChatCompletion(goldenContext(false), key, goldenRequest[schemas.BifrostChatRequest](t, p.GetProviderKey(), goldenClaudeChatNoLimit))
			checkGolden(t, err)
		}},
		{"claude chat stream", claudePath + "claude-sonnet-4-5:streamRawPredict", func(t *testing.T, p *VertexProvider, key schemas.Key) {
			drainGoldenStream(t)(p.ChatCompletionStream(goldenContext(false), goldenNoHooks, nil, key, goldenRequest[schemas.BifrostChatRequest](t, p.GetProviderKey(), goldenClaudeChat)))
		}},
		{"claude responses", claudeBetaPath + "claude-sonnet-4-5:rawPredict", func(t *testing.T, p *VertexProvider, key schemas.Key) {
			_, err := p.Responses(goldenContext(false), key, goldenRequest[schemas.BifrostResponsesRequest](t, p.GetProviderKey(), goldenClaudeResponses))
			checkGolden(t, err)
		}},
		{"claude responses stream", claudePath + "claude-sonnet-4-5:streamRawPredict", func(t *testing.T, p *VertexProvider, key schemas.Key) {
			drainGoldenStream(t)(p.ResponsesStream(goldenContext(false), goldenNoHooks, nil, key, goldenRequest[schemas.BifrostResponsesRequest](t, p.GetProviderKey(), goldenClaudeResponses)))
		}},
		{"claude route messages", claudePath + "claude-sonnet-4-5:rawPredict", passthrough(false, schemas.ClaudeRouteMessages, "", goldenClaudeRouteBody)},
		{"claude route messages stream", claudePath + "claude-sonnet-4-5:streamRawPredict", passthrough(true, schemas.ClaudeRouteMessages, "", goldenClaudeRouteBody)},
		{"passthrough", claudePath + "claude-sonnet-4-5:rawPredict", passthrough(false, "", claudePath+"claude-sonnet-4-5:rawPredict", goldenClaudeRouteBody)},
		{"passthrough stream", claudePath + "claude-sonnet-4-5:streamRawPredict", passthrough(true, "", claudePath+"claude-sonnet-4-5:streamRawPredict", goldenClaudeRouteBody)},
	}
}

// TestVertexCustomBaseMatchesNative is the custom-base golden: a custom provider on the vertex base
// sends exactly what native vertex sends for the same request and key (URL, headers, body), on
// every surface the router dispatches to it.
func TestVertexCustomBaseMatchesNative(t *testing.T) {
	// Capability rows exist for vertex only, so a family site that looks one up by the custom key
	// misses and sends something else.
	schemas.SetCapabilityResolver(func(provider schemas.ModelProvider, _ string) *schemas.ModelCapabilities {
		if provider == schemas.Vertex {
			return &schemas.ModelCapabilities{MaxOutputTokens: schemas.Ptr(1234)}
		}
		return nil
	})
	t.Cleanup(func() { schemas.SetCapabilityResolver(nil) })
	srv, seen := newVertexWireRecorder(t)
	native, nativeKey := newGoldenVertex(t, srv.URL, nil)
	custom, customKey := newGoldenVertex(t, srv.URL, &schemas.CustomProviderConfig{CustomProviderKey: string(customVertexKey), BaseProviderType: schemas.Vertex})
	if got := custom.GetProviderKey(); got != customVertexKey {
		t.Fatalf("custom GetProviderKey() = %q, want %q", got, customVertexKey)
	}
	if got := native.GetProviderKey(); got != schemas.Vertex {
		t.Fatalf("native GetProviderKey() = %q, want vertex", got)
	}
	for _, tc := range goldenCases() {
		t.Run(tc.name, func(t *testing.T) {
			seen.take()
			tc.call(t, native, nativeKey)
			want := seen.take()
			tc.call(t, custom, customKey)
			got := seen.take()
			if len(want) != 1 || want[0].Path != tc.path {
				t.Fatalf("native sent %+v, want one request to %q", want, tc.path)
			}
			if !strings.Contains(want[0].Body, goldenBodyHas[tc.name]) {
				t.Fatalf("native body %s lacks %s", want[0].Body, goldenBodyHas[tc.name])
			}
			if len(got) != 1 {
				t.Fatalf("custom sent %d requests, want 1", len(got))
			}
			if g, w := got[0], want[0]; !reflect.DeepEqual(g, w) {
				t.Fatalf("custom provider's wire request differs from native vertex\ncustom: %s %s?%s\n  %v\n  %s\nnative: %s %s?%s\n  %v\n  %s",
					g.Method, g.Path, g.Query, g.Header, g.Body, w.Method, w.Path, w.Query, w.Header, w.Body)
			}
		})
	}
}

// TestVertexCustomBaseListsUnderItsKey pins the identity sites: a custom provider on the vertex
// base lists its models as <key>/<model>, from the Model Garden catalog and from an API-key-only
// key's configuration alike.
func TestVertexCustomBaseListsUnderItsKey(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(vertexModelGarden(&hits, "gemini-2.5-flash"))
	t.Cleanup(srv.Close)
	custom, oauthKey := newGoldenVertex(t, srv.URL, &schemas.CustomProviderConfig{CustomProviderKey: string(customVertexKey), BaseProviderType: schemas.Vertex})
	oauthKey.Models = schemas.WhiteList{"*"}
	apiKey := schemas.Key{Value: *schemas.NewSecretVar("api-key"), Models: schemas.WhiteList{"gemini-2.5-pro"},
		VertexKeyConfig: &schemas.VertexKeyConfig{ProjectID: *schemas.NewSecretVar("proj-eu"), Region: *schemas.NewSecretVar("europe-west4")}}
	for name, tc := range map[string]struct {
		key  schemas.Key
		want string
	}{"model garden": {oauthKey, "vertex-eu/gemini-2.5-flash"}, "api-key configuration": {apiKey, "vertex-eu/gemini-2.5-pro"}} {
		t.Run(name, func(t *testing.T) {
			got := listedVertexModels(t, custom, tc.key)
			if _, ok := got[tc.want]; !ok || len(got) != 1 {
				t.Fatalf("listed %v, want only %s", got, tc.want)
			}
		})
	}
}

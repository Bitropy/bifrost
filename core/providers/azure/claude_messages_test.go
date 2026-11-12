package azure

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
)

// claudeMessagesTestBody carries the members a field-level rewrite could damage; it must reach
// the upstream byte for byte.
const claudeMessagesTestBody = `{"model":"my-deploy-7","max_tokens":64,"thinking":{"type":"enabled","budget_tokens":1024},` +
	`"mcp_servers":[{"type":"url","url":"https://mcp.example/sse","name":"m"}],` +
	`"tools":[{"type":"web_search_20260209","name":"web_search","max_uses":2}],` +
	`"messages":[{"role":"user","content":[{"type":"document","source":{"type":"text","media_type":"text/plain","data":"doc"},"citations":{"enabled":true},"cache_control":{"type":"ephemeral"}},` +
	`{"type":"text","text":"see projects/x/locations/y and \"models/z\" please"}]}],"x_unknown":{"a":  1}}`

type claudeUpstreamSeen struct {
	path, query, apiKey, auth, version, betas, body string
}

// TestAzureClaudeMessagesPassthrough drives both entry points against a fake Azure resource: the
// request goes to the Anthropic Messages endpoint with no api-version or client query, the body is
// untouched, anthropic-version is the client's or the default, betas are filtered for Azure, and
// Anthropic usage is parsed for a deployment name that does not look like a Claude model.
func TestAzureClaudeMessagesPassthrough(t *testing.T) {
	seen := make(chan claudeUpstreamSeen, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		seen <- claudeUpstreamSeen{r.URL.Path, r.URL.RawQuery, r.Header.Get("x-api-key"), r.Header.Get("Authorization"),
			r.Header.Get("anthropic-version"), r.Header.Get("anthropic-beta"), string(body)}
		if strings.Contains(string(body), `"stream":true`) {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":7,\"output_tokens\":1}}}\n\n"+
				"event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":5}}\n\n"+
				"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"type":"message","usage":{"input_tokens":7,"output_tokens":5}}`)
	}))
	t.Cleanup(srv.Close)

	provider, err := NewAzureProvider(&schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{
		DefaultRequestTimeoutInSeconds: 10,
		ExtraHeaders:                   map[string]schemas.SecretVar{"anthropic-beta": *schemas.NewSecretVar("context-1m-2025-08-07")},
	}}, &authTestLogger{})
	if err != nil {
		t.Fatalf("NewAzureProvider: %v", err)
	}
	key := schemas.Key{Value: *schemas.NewSecretVar("az-key"), AzureKeyConfig: &schemas.AzureKeyConfig{Endpoint: *schemas.NewSecretVar(srv.URL)}}

	streamBody := strings.Replace(claudeMessagesTestBody, `"max_tokens":64`, `"max_tokens":64,"stream":true`, 1)
	tests := []struct {
		name, betas, version, ctxToken string
		wantAPIKey, wantAuth           string
		wantVersion, wantBetas         string
		wantDropped                    []string
	}{
		{"api key, client version, betas filtered", "interleaved-thinking-2025-05-14,totally-unknown-2099", "2023-01-01", "",
			"az-key", "", "2023-01-01", "context-1m-2025-08-07,interleaved-thinking-2025-05-14", []string{"totally-unknown-2099"}},
		{"context bearer, default version, nothing dropped", "", "", "ctx-token",
			"", "Bearer ctx-token", AzureAnthropicAPIVersionDefault, "context-1m-2025-08-07", nil},
	}
	for _, tt := range tests {
		for _, stream := range []bool{false, true} {
			name := tt.name + "/unary"
			if stream {
				name = tt.name + "/stream"
			}
			t.Run(name, func(t *testing.T) {
				headers := map[string]string{"Content-Type": "application/json"}
				if tt.betas != "" {
					headers["Anthropic-Beta"] = tt.betas
				}
				if tt.version != "" {
					headers["anthropic-version"] = tt.version
				}
				body := claudeMessagesTestBody
				if stream {
					body = streamBody
				}
				req := &schemas.BifrostPassthroughRequest{
					Method: http.MethodPost, Path: "/openai/deployments/x/chat/completions", RawQuery: "api-version=2024-01-01&beta=true",
					Model: "my-deploy-7", Body: []byte(body), SafeHeaders: headers, ClaudeMessages: true,
				}
				ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
				if tt.ctxToken != "" {
					ctx.SetValue(AzureAuthorizationTokenKey, tt.ctxToken)
				}
				var dropped []string
				var usage *schemas.BifrostPassthroughUsage
				if stream {
					dropped, usage = drainClaudeStream(t, provider, ctx, key, req)
				} else {
					resp, bifrostErr := provider.Passthrough(ctx, key, req)
					if bifrostErr != nil {
						t.Fatalf("Passthrough: %s", bifrostErr.GetErrorString())
					}
					dropped, usage = resp.DroppedBetas, resp.PassthroughUsage
				}
				got := <-seen
				want := claudeUpstreamSeen{"/anthropic/v1/messages", "", tt.wantAPIKey, tt.wantAuth, tt.wantVersion, tt.wantBetas, body}
				if got != want {
					t.Fatalf("upstream saw\n %+v\nwant\n %+v", got, want)
				}
				if !reflect.DeepEqual(dropped, tt.wantDropped) {
					t.Fatalf("DroppedBetas = %#v, want %#v", dropped, tt.wantDropped)
				}
				if usage == nil || usage.LLMUsage == nil || usage.LLMUsage.PromptTokens != 7 || usage.LLMUsage.CompletionTokens != 5 {
					t.Fatalf("usage = %+v", usage)
				}
			})
		}
	}
}

func drainClaudeStream(t *testing.T, provider *AzureProvider, ctx *schemas.BifrostContext, key schemas.Key, req *schemas.BifrostPassthroughRequest) ([]string, *schemas.BifrostPassthroughUsage) {
	t.Helper()
	noHooks := func(_ *schemas.BifrostContext, resp *schemas.BifrostResponse, err *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError) {
		return resp, err
	}
	stream, bifrostErr := provider.PassthroughStream(ctx, noHooks, nil, key, req)
	if bifrostErr != nil {
		t.Fatalf("PassthroughStream: %s", bifrostErr.GetErrorString())
	}
	var dropped []string
	var usage *schemas.BifrostPassthroughUsage
	chunks := 0
	timeout := time.After(5 * time.Second)
	for {
		select {
		case chunk, ok := <-stream:
			if !ok {
				if chunks < 2 {
					t.Fatalf("got %d chunks, want data and a final chunk", chunks)
				}
				return dropped, usage
			}
			if chunk.BifrostError != nil {
				t.Fatalf("stream error: %s", chunk.BifrostError.GetErrorString())
			}
			pt := chunk.BifrostPassthroughResponse
			if chunks > 0 && !reflect.DeepEqual(pt.DroppedBetas, dropped) {
				t.Fatalf("chunk %d DroppedBetas %v, earlier %v", chunks, pt.DroppedBetas, dropped)
			}
			dropped = pt.DroppedBetas
			if pt.PassthroughUsage != nil {
				usage = pt.PassthroughUsage
			}
			chunks++
		case <-timeout:
			t.Fatal("stream did not finish")
		}
	}
}

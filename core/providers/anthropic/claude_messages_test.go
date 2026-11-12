package anthropic

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

const claudeMessagesTestBody = `{"model":"claude-x","max_tokens":64,"thinking":{"type":"enabled","budget_tokens":1024},` +
	`"mcp_servers":[{"type":"url","url":"https://mcp.example/sse","name":"m"}],` +
	`"tools":[{"type":"web_search_20260209","name":"web_search","max_uses":2}],` +
	`"messages":[{"role":"user","content":[{"type":"document","source":{"type":"text","media_type":"text/plain","data":"doc"},"citations":{"enabled":true},"cache_control":{"type":"ephemeral"}},` +
	`{"type":"text","text":"see projects/x/locations/y and \"models/z\" please"}]}],"x_unknown":{"a":  1}}`

func TestClaudeMessagesBetas(t *testing.T) {
	extra := map[string]schemas.SecretVar{"Anthropic-Beta": *schemas.NewSecretVar("context-1m-2025-08-07")}
	tests := []struct {
		name        string
		client      map[string]string
		provider    schemas.ModelProvider
		wantKept    []string
		wantDropped []string
	}{
		{"vertex drops unsupported and unknown", map[string]string{"anthropic-beta": " interleaved-thinking-2025-05-14,files-api-2025-04-14, x-unknown-1,context-1m-2025-08-07"}, schemas.Vertex,
			[]string{"context-1m-2025-08-07", "interleaved-thinking-2025-05-14"}, []string{"files-api-2025-04-14", "x-unknown-1"}},
		{"azure keeps files api", map[string]string{"ANTHROPIC-BETA": "files-api-2025-04-14"}, schemas.Azure,
			[]string{"context-1m-2025-08-07", "files-api-2025-04-14"}, nil},
		{"no client betas", nil, schemas.Vertex, []string{"context-1m-2025-08-07"}, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			kept, dropped := ClaudeMessagesBetas(tt.client, extra, tt.provider, nil)
			if !reflect.DeepEqual(kept, tt.wantKept) || !reflect.DeepEqual(dropped, tt.wantDropped) {
				t.Fatalf("kept %#v dropped %#v, want %#v %#v", kept, dropped, tt.wantKept, tt.wantDropped)
			}
		})
	}
	if kept, dropped := ClaudeMessagesBetas(nil, nil, schemas.Vertex, nil); kept != nil || dropped != nil {
		t.Fatalf("no betas: kept %#v dropped %#v", kept, dropped)
	}
}

// TestAnthropicClaudeMessagesPassthrough: the native provider sends a ClaudeMessages request to
// /v1/messages whatever Path says, keeps the client query, forwards the body and betas unchanged
// (an unknown beta included, nothing filtered or reported) and parses Messages usage.
func TestAnthropicClaudeMessagesPassthrough(t *testing.T) {
	type seen struct{ path, query, apiKey, betas, body string }
	got := make(chan seen, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		got <- seen{r.URL.Path, r.URL.RawQuery, r.Header.Get("x-api-key"), r.Header.Get("anthropic-beta"), string(body)}
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
	provider := NewAnthropicProvider(&schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{BaseURL: srv.URL}}, truncationTestLogger{})
	key := schemas.Key{Value: *schemas.NewSecretVar("sk-ant")}
	const betas = "interleaved-thinking-2025-05-14,totally-unknown-2099"

	for _, stream := range []bool{false, true} {
		name := "unary"
		body := claudeMessagesTestBody
		if stream {
			name = "stream"
			body = strings.Replace(body, `"max_tokens":64`, `"max_tokens":64,"stream":true`, 1)
		}
		t.Run(name, func(t *testing.T) {
			req := &schemas.BifrostPassthroughRequest{
				Method: http.MethodPost, Path: "/v1/models", RawQuery: "beta=true", Model: "claude-x",
				Body: []byte(body), SafeHeaders: map[string]string{"anthropic-beta": betas}, ClaudeMessages: true,
			}
			ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
			var dropped []string
			var usage *schemas.BifrostPassthroughUsage
			if stream {
				ch, bifrostErr := provider.PassthroughStream(ctx, truncationPassthroughPostHook, nil, key, req)
				if bifrostErr != nil {
					t.Fatalf("PassthroughStream: %s", bifrostErr.GetErrorString())
				}
				timeout := time.After(5 * time.Second)
				for done := false; !done; {
					select {
					case chunk, ok := <-ch:
						if !ok {
							done = true
							break
						}
						if chunk.BifrostError != nil {
							t.Fatalf("stream error: %s", chunk.BifrostError.GetErrorString())
						}
						if chunk.BifrostPassthroughResponse.DroppedBetas != nil {
							dropped = chunk.BifrostPassthroughResponse.DroppedBetas
						}
						if u := chunk.BifrostPassthroughResponse.PassthroughUsage; u != nil {
							usage = u
						}
					case <-timeout:
						t.Fatal("stream did not finish")
					}
				}
			} else {
				resp, bifrostErr := provider.Passthrough(ctx, key, req)
				if bifrostErr != nil {
					t.Fatalf("Passthrough: %s", bifrostErr.GetErrorString())
				}
				dropped, usage = resp.DroppedBetas, resp.PassthroughUsage
			}
			if g, want := <-got, (seen{"/v1/messages", "beta=true", "sk-ant", betas, body}); g != want {
				t.Fatalf("upstream saw\n %+v\nwant\n %+v", g, want)
			}
			if dropped != nil {
				t.Fatalf("DroppedBetas = %#v, want nil", dropped)
			}
			if usage == nil || usage.LLMUsage == nil || usage.LLMUsage.PromptTokens != 7 || usage.LLMUsage.CompletionTokens != 5 {
				t.Fatalf("usage = %+v", usage)
			}
		})
	}
}

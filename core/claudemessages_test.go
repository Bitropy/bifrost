package bifrost

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

// ClaudeMessages is refused with a 400 before any upstream call on a provider that does not
// declare it, on both entry points, and a custom provider inherits its base's answer. The
// Anthropic control proves the same request is served where it is declared.
func TestClaudeMessagesRefusedWhereUnsupported(t *testing.T) {
	const customOpenAI = schemas.ModelProvider("custom-openai")
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"type":"message","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	t.Cleanup(srv.Close)
	account := NewMockAccount()
	for _, provider := range []schemas.ModelProvider{schemas.OpenAI, schemas.Gemini, schemas.Anthropic, customOpenAI} {
		configureUpstream(account, provider, "key-"+string(provider), srv.URL, 0)
	}
	account.SetCustomProviderConfig(customOpenAI, &schemas.CustomProviderConfig{BaseProviderType: schemas.OpenAI})
	client, err := Init(context.Background(), schemas.BifrostConfig{Account: account, Logger: NewDefaultLogger(schemas.LogLevelError)})
	if err != nil {
		t.Fatalf("Init: %v", err)
	}
	t.Cleanup(client.Shutdown)
	newReq := func() *schemas.BifrostPassthroughRequest {
		return &schemas.BifrostPassthroughRequest{Method: http.MethodPost, Model: "claude-x", Body: []byte(`{"model":"claude-x"}`), ClaudeMessages: true}
	}

	for _, provider := range []schemas.ModelProvider{schemas.OpenAI, schemas.Gemini, customOpenAI} {
		t.Run(string(provider), func(t *testing.T) {
			_, unaryErr := client.Passthrough(newPassthroughContext(t), provider, newReq())
			_, streamErr := client.PassthroughStream(newPassthroughContext(t), provider, newReq())
			for name, bifrostErr := range map[string]*schemas.BifrostError{"Passthrough": unaryErr, "PassthroughStream": streamErr} {
				if bifrostErr == nil || bifrostErr.StatusCode == nil || *bifrostErr.StatusCode != http.StatusBadRequest {
					t.Fatalf("%s: want a 400, got %+v", name, bifrostErr)
				}
			}
			if hits.Load() != 0 {
				t.Fatal("the upstream was called")
			}
		})
	}

	// The model is spliced into Vertex's upstream path, so it must be one non-empty segment.
	for _, model := range []string{"", "x/../../endpoints/1", "m?key=1", "m#f", "m%2F", `m\n`} {
		t.Run("model "+model, func(t *testing.T) {
			req := func() *schemas.BifrostPassthroughRequest {
				r := newReq()
				r.Model = model
				return r
			}
			_, unaryErr := client.Passthrough(newPassthroughContext(t), schemas.Anthropic, req())
			_, streamErr := client.PassthroughStream(newPassthroughContext(t), schemas.Anthropic, req())
			for name, bifrostErr := range map[string]*schemas.BifrostError{"Passthrough": unaryErr, "PassthroughStream": streamErr} {
				if bifrostErr == nil || bifrostErr.StatusCode == nil || *bifrostErr.StatusCode != http.StatusBadRequest {
					t.Fatalf("%s: want a 400, got %+v", name, bifrostErr)
				}
			}
			if hits.Load() != 0 {
				t.Fatal("the upstream was called")
			}
		})
	}

	t.Run("anthropic control", func(t *testing.T) {
		resp, bifrostErr := client.Passthrough(newPassthroughContext(t), schemas.Anthropic, newReq())
		if bifrostErr != nil || resp.StatusCode != http.StatusOK || hits.Load() != 1 {
			t.Fatalf("resp %+v err %+v hits %d", resp, bifrostErr, hits.Load())
		}
	})
}

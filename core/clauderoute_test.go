package bifrost

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"sync/atomic"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

// TestValidateClaudeRoute loops every provider (each standard one, and custom providers built on
// anthropic and openai) over both entry points, every ClaudeRoute value, with and without Path, and
// checks validateClaudeRoute's answer against the BIT-2952 validation table. A rejected combination
// is also sent through the public entry point, which must answer the same and call no upstream.
func TestValidateClaudeRoute(t *testing.T) {
	const (
		customAnthropic = schemas.ModelProvider("custom-anthropic")
		customOpenAI    = schemas.ModelProvider("custom-openai")
	)
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"type":"message","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	t.Cleanup(srv.Close)
	providers := append(slices.Clone(schemas.StandardProviders), customAnthropic, customOpenAI)
	account := NewMockAccount()
	for _, provider := range providers {
		configureUpstream(account, provider, "key-"+string(provider), srv.URL, 0)
	}
	account.SetCustomProviderConfig(customAnthropic, &schemas.CustomProviderConfig{BaseProviderType: schemas.Anthropic})
	account.SetCustomProviderConfig(customOpenAI, &schemas.CustomProviderConfig{BaseProviderType: schemas.OpenAI})
	client, err := Init(context.Background(), schemas.BifrostConfig{Account: account, Logger: NewDefaultLogger(schemas.LogLevelError)})
	if err != nil {
		t.Fatalf("Init: %v", err)
	}
	t.Cleanup(client.Shutdown)

	base := map[schemas.ModelProvider]schemas.ModelProvider{customAnthropic: schemas.Anthropic, customOpenAI: schemas.OpenAI}
	// want is the spec's validation table: "" for valid, else "config" or "unsupported_operation".
	want := func(provider schemas.ModelProvider, route schemas.ClaudeRoute, path string, stream bool) string {
		b := provider
		if mapped, ok := base[provider]; ok {
			b = mapped
		}
		switch {
		case route == "":
			return ""
		case route != schemas.ClaudeRouteMessages && route != schemas.ClaudeRouteCountTokens:
			return "config"
		case path != "":
			return "config"
		case b != schemas.Anthropic && b != schemas.Vertex && b != schemas.Azure:
			return "config"
		case route == schemas.ClaudeRouteCountTokens && stream:
			return "config"
		case route == schemas.ClaudeRouteCountTokens && b == schemas.Azure:
			return "unsupported_operation"
		}
		return ""
	}
	kind := func(bifrostErr *schemas.BifrostError) string {
		switch {
		case bifrostErr == nil:
			return ""
		case bifrostErr.Error != nil && bifrostErr.Error.Code != nil && *bifrostErr.Error.Code == "unsupported_operation":
			return "unsupported_operation"
		case bifrostErr.StatusCode == nil && bifrostErr.Error != nil && bifrostErr.Error.Message != "":
			return "config"
		}
		return "unexpected: " + bifrostErr.GetErrorString()
	}

	routes := []schemas.ClaudeRoute{"", schemas.ClaudeRouteMessages, schemas.ClaudeRouteCountTokens, "bogus"}
	for _, provider := range providers {
		for _, route := range routes {
			for _, path := range []string{"", "/v1/messages"} {
				for _, stream := range []bool{false, true} {
					name := string(provider) + "/" + string(route) + "/path=" + path
					if stream {
						name += "/stream"
					}
					t.Run(name, func(t *testing.T) {
						req := func() *schemas.BifrostPassthroughRequest {
							return &schemas.BifrostPassthroughRequest{Method: http.MethodPost, Path: path, Model: "claude-x", Body: []byte(`{"model":"claude-x"}`), ClaudeRoute: route}
						}
						wantKind := want(provider, route, path, stream)
						if got := kind(client.validateClaudeRoute(provider, req(), stream)); got != wantKind {
							t.Fatalf("validateClaudeRoute = %q, want %q", got, wantKind)
						}
						if wantKind == "" {
							return
						}
						before := hits.Load()
						var bifrostErr *schemas.BifrostError
						if stream {
							_, bifrostErr = client.PassthroughStream(newPassthroughContext(t), provider, req())
						} else {
							_, bifrostErr = client.Passthrough(newPassthroughContext(t), provider, req())
						}
						if got := kind(bifrostErr); got != wantKind {
							t.Fatalf("entry point answered %q, want %q", got, wantKind)
						}
						if hits.Load() != before {
							t.Fatal("the upstream was called")
						}
					})
				}
			}
		}
	}

	t.Run("empty model", func(t *testing.T) {
		req := func() *schemas.BifrostPassthroughRequest {
			return &schemas.BifrostPassthroughRequest{Method: http.MethodPost, Body: []byte(`{}`), ClaudeRoute: schemas.ClaudeRouteMessages}
		}
		_, unaryErr := client.Passthrough(newPassthroughContext(t), schemas.Anthropic, req())
		_, streamErr := client.PassthroughStream(newPassthroughContext(t), schemas.Anthropic, req())
		for name, bifrostErr := range map[string]*schemas.BifrostError{"Passthrough": unaryErr, "PassthroughStream": streamErr} {
			if bifrostErr == nil || bifrostErr.StatusCode == nil || *bifrostErr.StatusCode != http.StatusBadRequest {
				t.Fatalf("%s: want a 400, got %+v", name, bifrostErr)
			}
		}
	})

	t.Run("anthropic control", func(t *testing.T) {
		before := hits.Load()
		req := &schemas.BifrostPassthroughRequest{Method: http.MethodPost, Model: "claude-x", Body: []byte(`{"model":"claude-x"}`), ClaudeRoute: schemas.ClaudeRouteMessages}
		resp, bifrostErr := client.Passthrough(newPassthroughContext(t), schemas.Anthropic, req)
		if bifrostErr != nil || resp.StatusCode != http.StatusOK || hits.Load() != before+1 {
			t.Fatalf("resp %+v err %+v hits %d", resp, bifrostErr, hits.Load()-before)
		}
	})
}

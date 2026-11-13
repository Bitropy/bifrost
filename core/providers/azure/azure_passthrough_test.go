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

func TestBuildPassthroughURL(t *testing.T) {
	t.Parallel()

	provider := &AzureProvider{}
	endpoint := "https://my-resource.openai.azure.com"

	makeKey := func(endpoint string) schemas.Key {
		return schemas.Key{
			AzureKeyConfig: &schemas.AzureKeyConfig{
				Endpoint: *schemas.NewSecretVar(endpoint),
			},
		}
	}

	tests := []struct {
		name     string
		path     string
		rawQuery string
		want     string
	}{
		// --- Path normalisation ---
		{
			name: "normalise /openai/responses to /openai/v1/responses",
			path: "/openai/responses",
			want: endpoint + "/openai/v1/responses",
		},
		{
			name: "normalise /openai/videos to /openai/v1/videos",
			path: "/openai/videos",
			want: endpoint + "/openai/v1/videos",
		},

		// --- /openai/deployments/ — inject default api-version when absent ---
		{
			name: "deployments route: inject default api-version when missing",
			path: "/openai/deployments/gpt-4o/chat/completions",
			want: endpoint + "/openai/deployments/gpt-4o/chat/completions?api-version=" + DefaultAzureAPIVersion,
		},
		{
			name:     "deployments route: preserve caller-supplied api-version",
			path:     "/openai/deployments/gpt-4o/chat/completions",
			rawQuery: "api-version=2024-10-21",
			want:     endpoint + "/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21",
		},
		{
			name:     "deployments route: preserve caller api-version alongside other params",
			path:     "/openai/deployments/gpt-4o/chat/completions",
			rawQuery: "api-version=2024-06-01&foo=bar",
			want:     endpoint + "/openai/deployments/gpt-4o/chat/completions?api-version=2024-06-01&foo=bar",
		},

		// --- /openai/v1/responses — no api-version injected by default ---
		{
			name: "responses route: no api-version injected when missing and no override configured",
			path: "/openai/v1/responses",
			want: endpoint + "/openai/v1/responses",
		},
		{
			name:     "responses route: preserve caller-supplied api-version",
			path:     "/openai/v1/responses",
			rawQuery: "api-version=2025-01-01-preview",
			want:     endpoint + "/openai/v1/responses?api-version=2025-01-01-preview",
		},
		{
			name: "responses/compact route: no api-version injected when missing",
			path: "/openai/v1/responses/compact",
			want: endpoint + "/openai/v1/responses/compact",
		},

		// --- Anthropic routes — strip api-version ---
		{
			name:     "anthropic route: strip api-version",
			path:     "/anthropic/v1/messages",
			rawQuery: "api-version=2024-10-21",
			want:     endpoint + "/anthropic/v1/messages",
		},
		{
			name:     "anthropic route: preserve other query params after strip",
			path:     "/anthropic/v1/messages",
			rawQuery: "api-version=preview&foo=bar",
			want:     endpoint + "/anthropic/v1/messages?foo=bar",
		},

		// --- /openai/v1/videos — strip api-version ---
		{
			name:     "videos route: strip api-version",
			path:     "/openai/v1/videos",
			rawQuery: "api-version=2024-10-21",
			want:     endpoint + "/openai/v1/videos",
		},

		// --- Other v1 routes — pass through unchanged ---
		{
			name: "v1 chat completions: no api-version injected",
			path: "/openai/v1/chat/completions",
			want: endpoint + "/openai/v1/chat/completions",
		},
		{
			name:     "v1 chat completions: caller params passed through unchanged",
			path:     "/openai/v1/chat/completions",
			rawQuery: "foo=bar",
			want:     endpoint + "/openai/v1/chat/completions?foo=bar",
		},
		{
			name: "v1 embeddings: no api-version injected",
			path: "/openai/v1/embeddings",
			want: endpoint + "/openai/v1/embeddings",
		},
		{
			name: "v1 audio transcriptions: no api-version injected",
			path: "/openai/v1/audio/transcriptions",
			want: endpoint + "/openai/v1/audio/transcriptions",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, _ := provider.buildPassthroughURL(nil, makeKey(endpoint), tt.path, tt.rawQuery)
			if got != tt.want {
				t.Errorf("\ngot:  %s\nwant: %s", got, tt.want)
			}
		})
	}
}

// TestBuildPassthroughURL_AliasAPIVersionOverride verifies that when the
// resolved alias carries an AzureAliasCfg.APIVersion override, it is attached
// to the request — as the route default for /deployments/ (DefaultAzureAPIVersion),
// or as an explicit opt-in for /openai/v1/responses (which otherwise omits
// api-version entirely) — only in the path where the caller did NOT supply
// api-version themselves. Caller-supplied wins over alias override; alias
// override wins over route default.
func TestBuildPassthroughURL_AliasAPIVersionOverride(t *testing.T) {
	t.Parallel()

	provider := &AzureProvider{}
	endpoint := "https://my-resource.openai.azure.com"
	makeKey := schemas.Key{
		AzureKeyConfig: &schemas.AzureKeyConfig{
			Endpoint: *schemas.NewSecretVar(endpoint),
		},
	}

	// Build a ctx carrying an alias with APIVersion override.
	overrideVer := "2024-10-21"
	ctx := schemas.NewBifrostContext(nil, schemas.NoDeadline)
	ctx.SetValue(schemas.BifrostContextKeyResolvedAlias, &schemas.ResolvedAlias{
		Key: "best-model",
		Config: &schemas.AliasConfig{
			ModelID: "gpt-4o-deployment",
			AzureAliasCfg: &schemas.AzureAliasCfg{
				APIVersion: &overrideVer,
			},
		},
	})

	t.Run("deployments route: alias APIVersion overrides default", func(t *testing.T) {
		got, _ := provider.buildPassthroughURL(ctx, makeKey, "/openai/deployments/gpt-4o/chat/completions", "")
		want := endpoint + "/openai/deployments/gpt-4o/chat/completions?api-version=" + overrideVer
		if got != want {
			t.Errorf("\ngot:  %s\nwant: %s", got, want)
		}
	})

	t.Run("responses route: alias APIVersion is attached as explicit opt-in", func(t *testing.T) {
		got, _ := provider.buildPassthroughURL(ctx, makeKey, "/openai/v1/responses", "")
		want := endpoint + "/openai/v1/responses?api-version=" + overrideVer
		if got != want {
			t.Errorf("\ngot:  %s\nwant: %s", got, want)
		}
	})

	t.Run("caller-supplied api-version wins over alias override", func(t *testing.T) {
		got, _ := provider.buildPassthroughURL(ctx, makeKey, "/openai/deployments/gpt-4o/chat/completions", "api-version=2023-01-01")
		want := endpoint + "/openai/deployments/gpt-4o/chat/completions?api-version=2023-01-01"
		if got != want {
			t.Errorf("\ngot:  %s\nwant: %s", got, want)
		}
	})
}

// TestResolveAPIVersion_NoAlias verifies the helper returns the route default
// when no resolved alias is in ctx (covers the legacy code path).
func TestResolveAPIVersion_NoAlias(t *testing.T) {
	if got := resolveAPIVersion(nil, DefaultAzureAPIVersion); got != DefaultAzureAPIVersion {
		t.Errorf("got %q, want %q", got, DefaultAzureAPIVersion)
	}
	ctx := schemas.NewBifrostContext(nil, schemas.NoDeadline)
	if got := resolveAPIVersion(ctx, AzureAPIVersionPreview); got != AzureAPIVersionPreview {
		t.Errorf("got %q, want %q", got, AzureAPIVersionPreview)
	}
}

// TestResolveAzureEndpoint_AliasOverride verifies the Endpoint override path.
// Lets one Azure credential cover deployments hosted on multiple cognitive-
// services resources.
func TestResolveAzureEndpoint_AliasOverride(t *testing.T) {
	keyEndpoint := "https://primary.openai.azure.com"
	aliasEndpoint := "https://anthropic-resource.openai.azure.com"
	key := schemas.Key{
		AzureKeyConfig: &schemas.AzureKeyConfig{
			Endpoint: *schemas.NewSecretVar(keyEndpoint),
		},
	}

	// No alias: falls back to key-level endpoint.
	if got := resolveAzureEndpoint(nil, key); got != keyEndpoint {
		t.Errorf("nil ctx: got %q, want key-level %q", got, keyEndpoint)
	}
	ctx := schemas.NewBifrostContext(nil, schemas.NoDeadline)
	if got := resolveAzureEndpoint(ctx, key); got != keyEndpoint {
		t.Errorf("empty ctx: got %q, want key-level %q", got, keyEndpoint)
	}

	// With alias-level Endpoint override, alias wins.
	override := schemas.NewSecretVar(aliasEndpoint)
	ctx.SetValue(schemas.BifrostContextKeyResolvedAlias, &schemas.ResolvedAlias{
		Key: "best-claude",
		Config: &schemas.AliasConfig{
			ModelID: "claude-deployment",
			AzureAliasCfg: &schemas.AzureAliasCfg{
				Endpoint: override,
			},
		},
	})
	if got := resolveAzureEndpoint(ctx, key); got != aliasEndpoint {
		t.Errorf("alias override: got %q, want %q", got, aliasEndpoint)
	}

	// Alias with empty Endpoint value falls through to key-level — guards against
	// a misconfigured alias accidentally erasing the endpoint.
	emptyOverride := schemas.NewSecretVar("")
	ctx2 := schemas.NewBifrostContext(nil, schemas.NoDeadline)
	ctx2.SetValue(schemas.BifrostContextKeyResolvedAlias, &schemas.ResolvedAlias{
		Key: "x",
		Config: &schemas.AliasConfig{
			ModelID: "x",
			AzureAliasCfg: &schemas.AzureAliasCfg{
				Endpoint: emptyOverride,
			},
		},
	})
	if got := resolveAzureEndpoint(ctx2, key); got != keyEndpoint {
		t.Errorf("empty alias endpoint should fall through: got %q, want %q", got, keyEndpoint)
	}
}

// TestResolveAnthropicVersion_AliasOverride verifies the AnthropicVersion
// override path mirrors the APIVersion behavior.
func TestResolveAnthropicVersion_AliasOverride(t *testing.T) {
	if got := resolveAnthropicVersion(nil); got != AzureAnthropicAPIVersionDefault {
		t.Errorf("nil ctx: got %q, want default", got)
	}
	override := "2024-10-22"
	ctx := schemas.NewBifrostContext(nil, schemas.NoDeadline)
	ctx.SetValue(schemas.BifrostContextKeyResolvedAlias, &schemas.ResolvedAlias{
		Key: "best-claude",
		Config: &schemas.AliasConfig{
			ModelID: "claude-deployment",
			AzureAliasCfg: &schemas.AzureAliasCfg{
				AnthropicVersion: &override,
			},
		},
	})
	if got := resolveAnthropicVersion(ctx); got != override {
		t.Errorf("got %q, want %q", got, override)
	}
}

// claudeRouteTestBody carries the members a field-level rewrite could damage; it must reach
// the upstream byte for byte.
const claudeRouteTestBody = `{"model":"my-deploy-7","max_tokens":64,"thinking":{"type":"enabled","budget_tokens":1024},` +
	`"mcp_servers":[{"type":"url","url":"https://mcp.example/sse","name":"m"}],` +
	`"tools":[{"type":"web_search_20260209","name":"web_search","max_uses":2}],` +
	`"messages":[{"role":"user","content":[{"type":"document","source":{"type":"text","media_type":"text/plain","data":"doc"},"citations":{"enabled":true},"cache_control":{"type":"ephemeral"}},` +
	`{"type":"text","text":"see projects/x/locations/y and \"models/z\" please"}]}],"x_unknown":{"a":  1}}`

type claudeUpstreamSeen struct {
	path, query, apiKey, auth, version, betas, body string
}

// TestAzureClaudeRoutePassthrough drives the Messages route on both entry points against a fake Azure resource: the
// request goes to the Anthropic Messages endpoint with no api-version or client query, the body is
// untouched, anthropic-version is the client's or the default, betas are filtered for Azure, and
// Anthropic usage is parsed for a deployment name that does not look like a Claude model.
func TestAzureClaudeRoutePassthrough(t *testing.T) {
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

	streamBody := strings.Replace(claudeRouteTestBody, `"max_tokens":64`, `"max_tokens":64,"stream":true`, 1)
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
				body := claudeRouteTestBody
				if stream {
					body = streamBody
				}
				req := &schemas.BifrostPassthroughRequest{
					Method: http.MethodPost, RawQuery: "api-version=2024-01-01&beta=true",
					Model: "my-deploy-7", Body: []byte(body), SafeHeaders: headers, ClaudeRoute: schemas.ClaudeRouteMessages,
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

// TestAzurePassthroughAliasedClaudeUsage pins that a plain (path-addressed) unary passthrough to a
// deployment whose resolved alias is a Claude model is billed by the Anthropic usage parser, the
// same family check its auth and stream path use.
func TestAzurePassthroughAliasedClaudeUsage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"type":"message","usage":{"input_tokens":7,"output_tokens":5}}`)
	}))
	t.Cleanup(srv.Close)
	provider, err := NewAzureProvider(&schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{DefaultRequestTimeoutInSeconds: 10}}, &authTestLogger{})
	if err != nil {
		t.Fatalf("NewAzureProvider: %v", err)
	}
	key := schemas.Key{Value: *schemas.NewSecretVar("az-key"), AzureKeyConfig: &schemas.AzureKeyConfig{Endpoint: *schemas.NewSecretVar(srv.URL)}}
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	ctx.SetValue(schemas.BifrostContextKeyResolvedAlias, &schemas.ResolvedAlias{Key: "my-deploy-7", Config: &schemas.AliasConfig{ModelID: "claude-sonnet-4-5"}})
	resp, bifrostErr := provider.Passthrough(ctx, key, &schemas.BifrostPassthroughRequest{
		Method: http.MethodPost, Path: "/anthropic/v1/messages", Model: "my-deploy-7",
		Body: []byte(`{"model":"my-deploy-7","messages":[]}`), SafeHeaders: map[string]string{"Content-Type": "application/json"},
	})
	if bifrostErr != nil {
		t.Fatalf("Passthrough: %s", bifrostErr.GetErrorString())
	}
	if u := resp.PassthroughUsage; u == nil || u.LLMUsage == nil || u.LLMUsage.PromptTokens != 7 || u.LLMUsage.CompletionTokens != 5 {
		t.Fatalf("usage = %+v, want the Anthropic parse 7/5", u)
	}
}

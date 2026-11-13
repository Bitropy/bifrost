package vertex

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
	"golang.org/x/oauth2"
)

// claudeRouteTestBody carries the members a regex or field-level rewrite could damage: a server
// tool, MCP servers, thinking, cache_control, a citations document, an unknown key with inner
// whitespace, and a prompt quoting a Vertex resource path and a short model name.
const claudeRouteTestBody = `{"model":"claude-x","max_tokens":64,"thinking":{"type":"enabled","budget_tokens":1024},` +
	`"mcp_servers":[{"type":"url","url":"https://mcp.example/sse","name":"m"}],` +
	`"tools":[{"type":"web_search_20260209","name":"web_search","max_uses":2}],` +
	`"messages":[{"role":"user","content":[{"type":"document","source":{"type":"text","media_type":"text/plain","data":"doc"},"citations":{"enabled":true},"cache_control":{"type":"ephemeral"}},` +
	`{"type":"text","text":"see projects/x/locations/y and \"models/z\" please"}]}],"x_unknown":{"a":  1}}`

// claudeRouteTestRest is claudeRouteTestBody without its model member.
var claudeRouteTestRest = strings.Replace(claudeRouteTestBody, `"model":"claude-x",`, "", 1)

func TestClaudeRouteBody(t *testing.T) {
	tests := []struct {
		name, body string
		betas      []string
		bodyModel  string
		want       string
	}{
		{"model removed, version and betas added", claudeRouteTestBody, []string{"a-1", "b-2"}, "",
			claudeRouteTestRest[:len(claudeRouteTestRest)-1] + `,"anthropic_version":"vertex-2023-10-16","anthropic_beta":["a-1","b-2"]}`},
		{"no betas, no member", claudeRouteTestBody, nil, "",
			claudeRouteTestRest[:len(claudeRouteTestRest)-1] + `,"anthropic_version":"vertex-2023-10-16"}`},
		{"existing version kept, client anthropic_beta replaced", ` {"anthropic_version":"v-own", "model":"m","anthropic_beta":["x"],"a":[1, 2]}`, []string{"k"}, "",
			`{"anthropic_version":"v-own","a":[1, 2],"anthropic_beta":["k"]}`},
		{"client anthropic_beta removed when none kept", `{"anthropic_beta":["x"],"a":1}`, nil, "",
			`{"a":1,"anthropic_version":"vertex-2023-10-16"}`},
		{"empty object", `{}`, nil, "", `{"anthropic_version":"vertex-2023-10-16"}`},
		{"invalid JSON unchanged", `{"model":"m",`, []string{"k"}, "", `{"model":"m",`},
		{"array unchanged", `[1]`, nil, "", `[1]`},
		{"count-tokens keeps model", claudeRouteTestBody, []string{"a-1"}, "claude-x",
			claudeRouteTestBody[:len(claudeRouteTestBody)-1] + `,"anthropic_version":"vertex-2023-10-16","anthropic_beta":["a-1"]}`},
		{"count-tokens model set in place", ` {"a":1, "model" : "tok__x","b":2}`, nil, "claude-sonnet-4-5@20250929",
			`{"a":1,"model":"claude-sonnet-4-5@20250929","b":2,"anthropic_version":"vertex-2023-10-16"}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := string(claudeRouteBody([]byte(tt.body), tt.betas, tt.bodyModel)); got != tt.want {
				t.Fatalf("got  %s\nwant %s", got, tt.want)
			}
		})
	}
}

type claudeUpstreamSeen struct {
	path, query, auth, betaHeader string
	body                          string
}

// TestVertexClaudeRoutePassthrough drives every route on its entry points against a fake Vertex:
// Messages goes to the Anthropic publisher's rawPredict (unary) or streamRawPredict (stream) for
// Model, count-tokens to the count-tokens model on Model's host. The body is adapted without touching
// any other member (count-tokens keeps model, which the endpoint reads), betas move from the header
// into the body filtered for Vertex, and Anthropic usage is parsed on Messages only.
func TestVertexClaudeRoutePassthrough(t *testing.T) {
	seen := make(chan claudeUpstreamSeen, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		seen <- claudeUpstreamSeen{r.URL.Path, r.URL.RawQuery, r.Header.Get("Authorization"), r.Header.Get("anthropic-beta"), string(body)}
		if strings.HasSuffix(r.URL.Path, ":streamRawPredict") {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":7,\"output_tokens\":1}}}\n\n"+
				"event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":5}}\n\n"+
				"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if strings.Contains(r.URL.Path, "/count-tokens:") {
			_, _ = io.WriteString(w, `{"input_tokens":7}`)
			return
		}
		_, _ = io.WriteString(w, `{"type":"message","usage":{"input_tokens":7,"output_tokens":5}}`)
	}))
	t.Cleanup(srv.Close)
	provider, key := newClaudeTestVertex(t, srv.URL)
	const upstreamPrefix = "/v1/projects/p/locations/us-east5/publishers/anthropic/models/"

	const versioned = `{"anthropic_version":"v-own","model":"claude-x","messages":[]}`
	tests := []struct {
		name, body, betas string
		route             schemas.ClaudeRoute
		model             string
		alias             *schemas.ResolvedAlias
		wantModel         string // the upstream model segment, before the method
		wantBody          string
		wantDropped       []string
	}{
		{"betas filtered", claudeRouteTestBody, "interleaved-thinking-2025-05-14, files-api-2025-04-14,totally-unknown-2099", schemas.ClaudeRouteMessages, "claude-x", nil, "claude-x",
			claudeRouteTestRest[:len(claudeRouteTestRest)-1] + `,"anthropic_version":"vertex-2023-10-16","anthropic_beta":["context-1m-2025-08-07","interleaved-thinking-2025-05-14"]}`,
			[]string{"files-api-2025-04-14", "totally-unknown-2099"}},
		{"version kept, nothing dropped", versioned, "", schemas.ClaudeRouteMessages, "claude-x", nil, "claude-x",
			`{"anthropic_version":"v-own","messages":[],"anthropic_beta":["context-1m-2025-08-07"]}`, nil},
		{"resolved alias names the model", versioned, "", schemas.ClaudeRouteMessages, "tok__claude-x",
			&schemas.ResolvedAlias{Key: "tok__claude-x", Config: &schemas.AliasConfig{ModelID: "claude-sonnet-4-5@20250929"}}, "claude-sonnet-4-5@20250929",
			`{"anthropic_version":"v-own","messages":[],"anthropic_beta":["context-1m-2025-08-07"]}`, nil},
		{"count-tokens keeps model, betas filtered", claudeRouteTestBody, "files-api-2025-04-14", schemas.ClaudeRouteCountTokens, "claude-x", nil, "count-tokens",
			claudeRouteTestBody[:len(claudeRouteTestBody)-1] + `,"anthropic_version":"vertex-2023-10-16","anthropic_beta":["context-1m-2025-08-07"]}`,
			[]string{"files-api-2025-04-14"}},
		{"count-tokens version kept", versioned, "", schemas.ClaudeRouteCountTokens, "claude-x", nil, "count-tokens",
			`{"anthropic_version":"v-own","model":"claude-x","messages":[],"anthropic_beta":["context-1m-2025-08-07"]}`, nil},
		{"count-tokens body names the resolved alias model", versioned, "", schemas.ClaudeRouteCountTokens, "tok__claude-x",
			&schemas.ResolvedAlias{Key: "tok__claude-x", Config: &schemas.AliasConfig{ModelID: "claude-sonnet-4-5@20250929"}}, "count-tokens",
			`{"anthropic_version":"v-own","model":"claude-sonnet-4-5@20250929","messages":[],"anthropic_beta":["context-1m-2025-08-07"]}`, nil},
	}
	for _, tt := range tests {
		for _, stream := range []bool{false, true} {
			if stream && tt.route == schemas.ClaudeRouteCountTokens {
				continue // core refuses a count-tokens stream
			}
			name := tt.name + "/unary"
			if stream {
				name = tt.name + "/stream"
			}
			t.Run(name, func(t *testing.T) {
				headers := map[string]string{"Content-Type": "application/json"}
				if tt.betas != "" {
					headers["anthropic-beta"] = tt.betas
				}
				req := &schemas.BifrostPassthroughRequest{
					Method: http.MethodPost, RawQuery: "beta=true", Model: tt.model,
					Body: []byte(tt.body), SafeHeaders: headers, ClaudeRoute: tt.route,
				}
				ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
				if tt.alias != nil {
					ctx.SetValue(schemas.BifrostContextKeyResolvedAlias, tt.alias)
				}
				var dropped []string
				var usage *schemas.BifrostPassthroughUsage
				method := ":rawPredict"
				messages := tt.route == schemas.ClaudeRouteMessages
				if stream {
					method = ":streamRawPredict"
					dropped, usage = drainClaudeStream(t, provider, ctx, key, req)
				} else {
					resp, bifrostErr := provider.Passthrough(ctx, key, req)
					if bifrostErr != nil {
						t.Fatalf("Passthrough: %s", bifrostErr.GetErrorString())
					}
					dropped, usage = resp.DroppedBetas, resp.PassthroughUsage
					if want := upstreamPrefix + tt.wantModel + method; resp.ExtraFields.PassthroughPath != want {
						t.Fatalf("PassthroughPath = %q, want the upstream path %q", resp.ExtraFields.PassthroughPath, want)
					}
				}
				got := <-seen
				want := claudeUpstreamSeen{upstreamPrefix + tt.wantModel + method, "", "Bearer t", "", tt.wantBody}
				if got != want {
					t.Fatalf("upstream saw\n %+v\nwant\n %+v", got, want)
				}
				if !reflect.DeepEqual(dropped, tt.wantDropped) {
					t.Fatalf("DroppedBetas = %#v, want %#v", dropped, tt.wantDropped)
				}
				if !messages {
					if usage != nil {
						t.Fatalf("count-tokens parsed usage %+v", usage)
					}
					return
				}
				if usage == nil || usage.LLMUsage == nil || usage.LLMUsage.PromptTokens != 7 || usage.LLMUsage.CompletionTokens != 5 {
					t.Fatalf("usage = %+v", usage)
				}
			})
		}
	}
}

// newClaudeTestVertex returns a Vertex provider on baseURL with a context-1m ExtraHeaders beta, and
// an OAuth key whose Value must not switch an Anthropic publisher call to the API-key query.
func newClaudeTestVertex(t *testing.T, baseURL string) (*VertexProvider, schemas.Key) {
	t.Helper()
	config := &schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{
		BaseURL: baseURL, AllowPrivateNetwork: true, DefaultRequestTimeoutInSeconds: 5,
		ExtraHeaders: map[string]schemas.SecretVar{"Anthropic-Beta": *schemas.NewSecretVar("context-1m-2025-08-07")},
	}}
	config.CheckAndSetDefaults()
	provider, err := NewVertexProvider(config, vertexEndpointTestLogger{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(provider.client.CloseIdleConnections)
	key := schemas.Key{Value: *schemas.NewSecretVar("api-key"), VertexKeyConfig: &schemas.VertexKeyConfig{
		ProjectID: *schemas.NewSecretVar("p"), Region: *schemas.NewSecretVar("us-east5"), AuthCredentials: *schemas.NewSecretVar("test-credentials"),
	}}
	provider.tokenSources.Store(getClientKey(vertexCredentialIdentity(key)), oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "t"}))
	return provider, key
}

// TestVertexAnthropicPassthroughStreamBillsContainer pins that a generic (path-addressed) Anthropic
// stream on Vertex hands the request body to the usage tracker, so a container the stream started
// is billed as one session.
func TestVertexAnthropicPassthroughStreamBillsContainer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":7,\"output_tokens\":1}}}\n\n"+
			"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"container\":{\"id\":\"c1\"}},\"usage\":{\"output_tokens\":5}}\n\n"+
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	}))
	t.Cleanup(srv.Close)
	provider, key := newClaudeTestVertex(t, srv.URL)
	req := &schemas.BifrostPassthroughRequest{
		Method: http.MethodPost, Path: "/v1/projects/p/locations/us-east5/publishers/anthropic/models/claude-x:streamRawPredict",
		Body: []byte(`{"anthropic_version":"vertex-2023-10-16","messages":[]}`), SafeHeaders: map[string]string{"Content-Type": "application/json"},
	}
	_, usage := drainClaudeStream(t, provider, schemas.NewBifrostContext(context.Background(), schemas.NoDeadline), key, req)
	if usage == nil || usage.LLMUsage == nil || usage.LLMUsage.ToolUsage == nil || usage.LLMUsage.ToolUsage.ContainerSessions[schemas.ContainerTierDefault] != 1 {
		t.Fatalf("usage = %+v, want one container session", usage)
	}
}

// drainClaudeStream reads a ClaudeRoute stream, checks every chunk carries the same
// DroppedBetas, and returns them with the final chunk's usage.
func drainClaudeStream(t *testing.T, provider *VertexProvider, ctx *schemas.BifrostContext, key schemas.Key, req *schemas.BifrostPassthroughRequest) ([]string, *schemas.BifrostPassthroughUsage) {
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

// TestVertexClaudeRouteRefusesUnsafeModel: the model Vertex splices into the URL (path or host), the
// request's or the resolved alias's, must be one safe segment; anything else is a 400 sent nowhere.
func TestVertexClaudeRouteRefusesUnsafeModel(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { hits.Add(1) }))
	t.Cleanup(srv.Close)
	provider, key := newClaudeTestVertex(t, srv.URL)
	tests := []struct {
		name, model string
		aliasModel  string
	}{
		{"path traversal", "x/../../endpoints/1", ""},
		{"query", "m?key=1", ""},
		{"fragment", "m#f", ""},
		{"escaped slash", "m%2F", ""},
		{"backslash", `m\x`, ""},
		{"newline", "m\n", ""},
		{"space", "m x", ""},
		{"resolved alias model", "claude-x", "x/../../endpoints/1"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
			if tt.aliasModel != "" {
				ctx.SetValue(schemas.BifrostContextKeyResolvedAlias, &schemas.ResolvedAlias{Key: tt.model, Config: &schemas.AliasConfig{ModelID: tt.aliasModel}})
			}
			req := func(route schemas.ClaudeRoute) *schemas.BifrostPassthroughRequest {
				return &schemas.BifrostPassthroughRequest{Method: http.MethodPost, Model: tt.model, Body: []byte(`{}`), ClaudeRoute: route}
			}
			_, unaryErr := provider.Passthrough(ctx, key, req(schemas.ClaudeRouteMessages))
			_, streamErr := provider.PassthroughStream(ctx, nil, nil, key, req(schemas.ClaudeRouteMessages))
			_, countErr := provider.Passthrough(ctx, key, req(schemas.ClaudeRouteCountTokens))
			for name, bifrostErr := range map[string]*schemas.BifrostError{"Passthrough": unaryErr, "PassthroughStream": streamErr, "count-tokens": countErr} {
				if bifrostErr == nil || bifrostErr.StatusCode == nil || *bifrostErr.StatusCode != http.StatusBadRequest {
					t.Fatalf("%s: want a 400, got %+v", name, bifrostErr)
				}
			}
			if hits.Load() != 0 {
				t.Fatal("the upstream was called")
			}
		})
	}
}

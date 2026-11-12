package vertex

import (
	"context"
	"encoding/json"
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"golang.org/x/oauth2"
)

// TestVertexBaseURLReplacesOrigin covers every AI Platform URL builder (BIT-2956 BL-2, BL-3):
// a configured base replaces scheme and host, and the path, region and project stay.
func TestVertexBaseURLReplacesOrigin(t *testing.T) {
	providerUtils.SetCapabilityResolver(func(provider schemas.ModelProvider, model string) *schemas.ModelCapabilities {
		if provider == schemas.Vertex && model == "claude-opus-4-7" {
			return &schemas.ModelCapabilities{IsVertexMultiRegionOnly: schemas.Ptr(true)}
		}
		return nil
	})
	t.Cleanup(func() { providerUtils.SetCapabilityResolver(nil) })

	// The trailing slash is trimmed.
	const base = "http://127.0.0.1:9/"
	const origin = "http://127.0.0.1:9"
	key := schemas.Key{VertexKeyConfig: &schemas.VertexKeyConfig{ProjectID: *schemas.NewSecretVar("p"), Region: *schemas.NewSecretVar("us-central1")}}
	must := func(u string, err *schemas.BifrostError) string {
		if err != nil {
			t.Fatalf("unexpected error: %v", err.GetErrorString())
		}
		return u
	}
	openAI := func(method schemas.VertexEndpointMethod) string {
		ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
		return must(openAIEndpointURL(ctx, base, key, &schemas.VertexEndpoint{EndpointID: "2222", Format: schemas.VertexEndpointOpenAIChat, Method: method}, false))
	}

	tests := []struct {
		name string
		got  func() string
		want string
	}{
		{"api base (pass-through)", func() string { return getVertexAPIBaseURL(base, "us-central1", "v1beta1") }, origin + "/v1beta1"},
		{"model-aware api base (count-tokens)", func() string {
			return getVertexModelAwareAPIBaseURL(base, "us-central1", "v1", "claude-opus-4-7", false, nil)
		}, origin + "/v1"},
		{"project location", func() string { return getVertexProjectLocationURL(base, "europe-central2", "v1", "p") }, origin + "/v1/projects/p/locations/europe-central2"},
		{"publisher model rawPredict", func() string {
			return getVertexPublisherModelURL(base, "us-central1", "v1", "p", "mistralai", "m", ":rawPredict")
		}, origin + "/v1/projects/p/locations/us-central1/publishers/mistralai/models/m:rawPredict"},
		{"model-aware publisher model keeps the pool promotion in the path", func() string {
			return getVertexModelAwarePublisherModelURL(base, "us-central1", "v1", "p", "anthropic", "claude-opus-4-7", ":rawPredict", false, nil)
		}, origin + "/v1/projects/p/locations/us/publishers/anthropic/models/claude-opus-4-7:rawPredict"},
		{"model-aware publisher model, regular model", func() string {
			return getVertexModelAwarePublisherModelURL(base, "us-central1", "v1", "p", "anthropic", "m", ":streamRawPredict", false, nil)
		}, origin + "/v1/projects/p/locations/us-central1/publishers/anthropic/models/m:streamRawPredict"},
		{"endpoint", func() string {
			return getVertexEndpointURL(base, "us-central1", "v1beta1", "p", "openapi/chat/completions", "")
		}, origin + "/v1beta1/projects/p/locations/us-central1/endpoints/openapi/chat/completions"},
		{"gemini publisher model", func() string {
			return getCompleteURLForGeminiEndpoint(base, "gemini-2.5-flash", "global", "p", "123", ":streamGenerateContent")
		}, origin + "/v1/projects/p/locations/global/publishers/google/models/gemini-2.5-flash:streamGenerateContent"},
		{"gemini tuned endpoint", func() string {
			return getCompleteURLForGeminiEndpoint(base, "4567", "us-central1", "p", "123", ":generateContent")
		}, origin + "/v1beta1/projects/123/locations/us-central1/endpoints/4567:generateContent"},
		{"model list", func() string { return getVertexModelListURL(base, "us", "google", 1000) }, origin + "/v1beta1/publishers/google/models?pageSize=1000"},
		{"cached contents", func() string { return vertexCachedContentBaseURL(base, "us-central1", "p") }, origin + "/v1/projects/p/locations/us-central1/cachedContents"},
		{"batch jobs base", func() string { return must(vertexBatchJobsBaseURL(base, key)) }, origin + "/v1/projects/p/locations/us-central1"},
		{"batch job, bare id", func() string { return must(vertexBatchJobURL(base, key, "123")) }, origin + "/v1/projects/p/locations/us-central1/batchPredictionJobs/123"},
		{"batch job, full resource name", func() string {
			return must(vertexBatchJobURL(base, key, "projects/987654321/locations/europe-west4/batchPredictionJobs/123"))
		}, origin + "/v1/projects/987654321/locations/europe-west4/batchPredictionJobs/123"},
		{"video operation, full resource name", func() string {
			return must(vertexVideoOperationURL(base, "projects/proj/locations/europe-west4/publishers/google/models/veo-3/operations/op-1"))
		}, origin + "/v1/projects/proj/locations/europe-west4/publishers/google/models/veo-3:fetchPredictOperation"},
		{"openai endpoint rawPredict", func() string { return openAI(schemas.VertexEndpointRawPredict) }, origin + "/v1/projects/p/locations/us-central1/endpoints/2222:rawPredict"},
		{"openai endpoint chat completions", func() string { return openAI(schemas.VertexEndpointChatCompletions) }, origin + "/v1beta1/projects/p/locations/us-central1/endpoints/2222/chat/completions"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.got(); got != tt.want {
				t.Fatalf("got %q, want %q", got, tt.want)
			}
		})
	}
}

// TestVertexBaseURLEmptyKeepsGoogleHosts pins the builders the host tables in utils_test.go do not reach (BL-1).
func TestVertexBaseURLEmptyKeepsGoogleHosts(t *testing.T) {
	t.Parallel()
	for _, tt := range []struct{ got, want string }{
		{getVertexModelListURL("", "us", "google", 1000), "https://aiplatform.googleapis.com/v1beta1/publishers/google/models?pageSize=1000"},
		{getVertexModelListURL("", "europe-central2", "anthropic", 1000), "https://europe-central2-aiplatform.googleapis.com/v1beta1/publishers/anthropic/models?pageSize=1000"},
		{getVertexAPIBaseURL("", "eu", "v1"), "https://aiplatform.eu.rep.googleapis.com/v1"},
		{getVertexModelAwareAPIBaseURL("", "europe-central2", "v1", "m", false, nil), "https://europe-central2-aiplatform.googleapis.com/v1"},
		{vertexCachedContentBaseURL("", "global", "p"), "https://aiplatform.googleapis.com/v1/projects/p/locations/global/cachedContents"},
	} {
		if tt.got != tt.want {
			t.Errorf("got %q, want %q", tt.got, tt.want)
		}
	}
}

// TestVertexBaseURLLeavesOtherGoogleHosts checks that rerank and GCS keep their own hosts
// when a base is set: the first dial is recorded and refused, so nothing leaves the test.
func TestVertexBaseURLLeavesOtherGoogleHosts(t *testing.T) {
	config := &schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{BaseURL: "http://127.0.0.1:9", DefaultRequestTimeoutInSeconds: 5}}
	config.CheckAndSetDefaults()
	provider, err := NewVertexProvider(config, vertexEndpointTestLogger{})
	if err != nil {
		t.Fatal(err)
	}
	var mu sync.Mutex
	var dialed []string
	provider.client.Dial = func(addr string) (net.Conn, error) {
		mu.Lock()
		dialed = append(dialed, addr)
		mu.Unlock()
		return nil, errors.New("dial refused by test")
	}
	key := schemas.Key{VertexKeyConfig: &schemas.VertexKeyConfig{
		ProjectID: *schemas.NewSecretVar("p"), Region: *schemas.NewSecretVar("us-central1"), AuthCredentials: *schemas.NewSecretVar("test-credentials"),
	}}
	provider.tokenSources.Store(getClientKey(vertexCredentialIdentity(key)), oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "t"}))
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	_, _ = provider.Rerank(ctx, key, &schemas.BifrostRerankRequest{Model: "semantic-ranker-default@latest", Query: "q", Documents: []schemas.RerankDocument{{Text: "d"}}})
	if _, err := provider.fetchGCSObjectEncoded(ctx, key, "gs://bucket/object"); err == nil {
		t.Fatal("GCS fetch succeeded through a refused dial")
	}

	mu.Lock()
	defer mu.Unlock()
	want := []string{"discoveryengine.googleapis.com:443", "storage.googleapis.com:443"}
	if strings.Join(dialed, ",") != strings.Join(want, ",") {
		t.Fatalf("dialed %v, want %v", dialed, want)
	}
}

// TestVertexBaseURLAuthorizedUserTokenRoundTrip runs the auth of a real credential JSON against
// a fake (BL-5): the authorized_user token_uri serves the token, and the managed and pass-through
// requests reach the base carrying it. A key with a Value sends ?key= to the base instead.
func TestVertexBaseURLAuthorizedUserTokenRoundTrip(t *testing.T) {
	type seen struct{ path, query, auth, host string }
	requests := make(chan seen, 4)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/token" {
			_, _ = io.WriteString(w, `{"access_token":"fake-token","token_type":"Bearer","expires_in":3600}`)
			return
		}
		requests <- seen{r.URL.Path, r.URL.RawQuery, r.Header.Get("Authorization"), r.Host}
		_, _ = io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`)
	}))
	t.Cleanup(srv.Close)

	config := &schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{BaseURL: srv.URL, AllowPrivateNetwork: true, DefaultRequestTimeoutInSeconds: 5}}
	config.CheckAndSetDefaults()
	provider, err := NewVertexProvider(config, vertexEndpointTestLogger{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(provider.client.CloseIdleConnections)
	creds, _ := json.Marshal(map[string]string{
		"type": "authorized_user", "client_id": "c", "client_secret": "s", "refresh_token": "r", "token_uri": srv.URL + "/token",
	})
	key := schemas.Key{VertexKeyConfig: &schemas.VertexKeyConfig{
		ProjectID: *schemas.NewSecretVar("p"), Region: *schemas.NewSecretVar("us-central1"), AuthCredentials: *schemas.NewSecretVar(string(creds)),
	}}
	srvURL, _ := url.Parse(srv.URL)
	const model = "/v1/projects/p/locations/us-central1/publishers/google/models/gemini-2.5-flash"
	check := func(t *testing.T, path, query, auth string) {
		t.Helper()
		// The handler records before it answers, so a returned call has left its record.
		select {
		case got := <-requests:
			if want := (seen{path, query, auth, srvURL.Host}); got != want {
				t.Fatalf("upstream saw %+v, want %+v", got, want)
			}
		default:
			t.Fatal("no request reached the base")
		}
	}
	chat := func(t *testing.T, key schemas.Key) {
		t.Helper()
		ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
		text := "hello"
		_, bifrostErr := provider.ChatCompletion(ctx, key, &schemas.BifrostChatRequest{
			Provider: schemas.Vertex, Model: "gemini-2.5-flash",
			Input: []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: &text}}},
		})
		if bifrostErr != nil {
			t.Fatalf("ChatCompletion: %s", bifrostErr.GetErrorString())
		}
	}

	t.Run("managed", func(t *testing.T) {
		chat(t, key)
		check(t, model+":generateContent", "", "Bearer fake-token")
	})
	t.Run("managed with api key", func(t *testing.T) {
		apiKey := key
		apiKey.Value = *schemas.NewSecretVar("api-key")
		chat(t, apiKey)
		check(t, model+":generateContent", "key=api-key", "")
	})
	t.Run("pass-through", func(t *testing.T) {
		ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
		_, bifrostErr := provider.Passthrough(ctx, key, &schemas.BifrostPassthroughRequest{
			Method: http.MethodPost, Path: "/publishers/google/models/gemini-2.5-flash:generateContent", Body: []byte(`{"contents":[]}`),
		})
		if bifrostErr != nil {
			t.Fatalf("Passthrough: %s", bifrostErr.GetErrorString())
		}
		check(t, model+":generateContent", "", "Bearer fake-token")
	})
	t.Run("pass-through stream", func(t *testing.T) {
		ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
		noHooks := func(_ *schemas.BifrostContext, resp *schemas.BifrostResponse, err *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError) {
			return resp, err
		}
		stream, bifrostErr := provider.PassthroughStream(ctx, noHooks, nil, key, &schemas.BifrostPassthroughRequest{
			Method: http.MethodPost, Path: "/publishers/google/models/gemini-2.5-flash:streamGenerateContent", Body: []byte(`{"contents":[]}`),
		})
		if bifrostErr != nil {
			t.Fatalf("PassthroughStream: %s", bifrostErr.GetErrorString())
		}
		for chunk := range stream {
			if chunk.BifrostError != nil {
				t.Fatalf("PassthroughStream chunk: %s", chunk.BifrostError.GetErrorString())
			}
		}
		check(t, model+":streamGenerateContent", "", "Bearer fake-token")
	})
}

// vertexHostFuncs are the only functions that may spell an AI Platform host (BL-4).
var vertexHostFuncs = map[string]bool{
	"getVertexAPIHost":             true,
	"getVertexModelAwareAPIHost":   true,
	"getVertexModelListingAPIHost": true,
}

// vertexURLViolations returns the position of each AI Platform host or https://%s literal
// outside the host functions, and of each host-function call outside a function that passes
// its host through vertexOrigin, so every URL honours a configured base. Comments are not literals.
func vertexURLViolations(fset *token.FileSet, files []*ast.File) []string {
	calls := func(decl ast.Decl, names map[string]bool, visit func(*ast.CallExpr)) {
		ast.Inspect(decl, func(n ast.Node) bool {
			if call, ok := n.(*ast.CallExpr); ok {
				if id, ok := call.Fun.(*ast.Ident); ok && names[id.Name] {
					visit(call)
				}
			}
			return true
		})
	}
	builders := map[string]bool{}
	for _, f := range files {
		for _, decl := range f.Decls {
			if fn, ok := decl.(*ast.FuncDecl); ok {
				calls(fn, map[string]bool{"vertexOrigin": true}, func(*ast.CallExpr) { builders[fn.Name.Name] = true })
			}
		}
	}
	var out []string
	for _, f := range files {
		for _, decl := range f.Decls {
			name := ""
			if fn, ok := decl.(*ast.FuncDecl); ok {
				name = fn.Name.Name
			}
			if vertexHostFuncs[name] {
				continue
			}
			ast.Inspect(decl, func(n ast.Node) bool {
				lit, ok := n.(*ast.BasicLit)
				if ok && lit.Kind == token.STRING && (strings.Contains(lit.Value, "aiplatform") || strings.Contains(lit.Value, "https://%s")) {
					out = append(out, fset.Position(lit.Pos()).String()+" literal in "+name)
				}
				return true
			})
			if !builders[name] {
				calls(decl, vertexHostFuncs, func(call *ast.CallExpr) {
					out = append(out, fset.Position(call.Pos()).String()+" host call in "+name)
				})
			}
		}
	}
	return out
}

func TestVertexURLLiteralGuard(t *testing.T) {
	t.Parallel()
	fset := token.NewFileSet()
	paths, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	var files []*ast.File
	for _, p := range paths {
		if strings.HasSuffix(p, "_test.go") {
			continue
		}
		src, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		f, err := parser.ParseFile(fset, p, src, parser.SkipObjectResolution)
		if err != nil {
			t.Fatal(err)
		}
		files = append(files, f)
	}
	if len(files) == 0 {
		t.Fatal("no source files scanned")
	}
	if v := vertexURLViolations(fset, files); len(v) > 0 {
		t.Fatalf("AI Platform URL literals outside the base-aware builders: %v", v)
	}

	planted, err := parser.ParseFile(fset, "planted.go", `package vertex
// https://us-central1-aiplatform.googleapis.com in a comment is fine
func plantedURL(region string) string { return "https://" + region + "-aiplatform.googleapis.com/v1" }
var plantedOrigin = fmt.Sprintf("https://%s", "x")
func plantedConcat(r string) string { return "https://" + getVertexAPIHost(r) + "/v1" }
func plantedCall(r string) string { return fmt.Sprint(getVertexModelListingAPIHost(r)) }
`, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	if v := vertexURLViolations(fset, []*ast.File{planted}); len(v) != 4 {
		t.Fatalf("planted: got %v, want 4 violations", v)
	}
}

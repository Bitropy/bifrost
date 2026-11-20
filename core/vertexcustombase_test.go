package bifrost

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

const customVertex = schemas.ModelProvider("vertex-eu")

type vertexCoreSeen struct{ path, query, body string }

// newVertexCustomBaseClient runs native vertex and a custom vertex-eu on the vertex base against
// one fake Vertex, both with the same API key so Gemini needs no OAuth exchange.
func newVertexCustomBaseClient(t *testing.T) (*Bifrost, func() []vertexCoreSeen) {
	t.Helper()
	var mu sync.Mutex
	var seen []vertexCoreSeen
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		seen = append(seen, vertexCoreSeen{r.URL.Path, r.URL.RawQuery, string(body)})
		mu.Unlock()
		const resp = `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`
		if strings.HasSuffix(r.URL.Path, ":streamGenerateContent") {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "data: "+resp+"\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, resp)
	}))
	t.Cleanup(srv.Close)
	account := NewMockAccount()
	for _, provider := range []schemas.ModelProvider{schemas.Vertex, customVertex} {
		account.AddProviderWithBaseURL(provider, 1, 1, srv.URL)
		account.configs[provider].NetworkConfig.AllowPrivateNetwork = true
		account.configs[provider].NetworkConfig.MaxRetries = 0
		account.SetKeysForProvider(provider, []schemas.Key{{
			ID: "key-" + string(provider), Value: *schemas.NewSecretVar("api-key"), Models: schemas.WhiteList{"*"}, Weight: 100,
			VertexKeyConfig: &schemas.VertexKeyConfig{ProjectID: *schemas.NewSecretVar("proj-eu"), Region: *schemas.NewSecretVar("europe-west4")},
		}})
	}
	account.SetCustomProviderConfig(customVertex, &schemas.CustomProviderConfig{BaseProviderType: schemas.Vertex})
	client, err := Init(t.Context(), schemas.BifrostConfig{Account: account, Logger: NewDefaultLogger(schemas.LogLevelError)})
	if err != nil {
		t.Fatalf("Init: %v", err)
	}
	t.Cleanup(client.Shutdown)
	take := func() []vertexCoreSeen {
		mu.Lock()
		defer mu.Unlock()
		out := seen
		seen = nil
		return out
	}
	return client, take
}

// TestVertexCustomBaseEndToEnd sends the same chat request through the public entry points to
// native vertex and to vertex-eu: vertex-eu is its own provider (getProviderByKey, response
// provider) and puts the same request on the wire, with its key's project and region in the path.
func TestVertexCustomBaseEndToEnd(t *testing.T) {
	client, take := newVertexCustomBaseClient(t)
	if got := client.getProviderByKey(customVertex); got == nil || got.GetProviderKey() != customVertex {
		t.Fatalf("getProviderByKey(%s) = %v", customVertex, got)
	}
	const wantPath = "/v1/projects/proj-eu/locations/europe-west4/publishers/google/models/gemini-2.5-flash:"
	chat := func(provider schemas.ModelProvider) *schemas.BifrostChatRequest {
		return &schemas.BifrostChatRequest{Provider: provider, Model: "gemini-2.5-flash", Input: []schemas.ChatMessage{{
			Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("hi")},
		}}}
	}
	for _, stream := range []bool{false, true} {
		name := "unary"
		if stream {
			name = "stream"
		}
		t.Run(name, func(t *testing.T) {
			send := func(provider schemas.ModelProvider) (schemas.ModelProvider, []vertexCoreSeen) {
				take()
				if !stream {
					resp, bifrostErr := client.ChatCompletionRequest(newPassthroughContext(t), chat(provider))
					if bifrostErr != nil {
						t.Fatalf("%s: %s", provider, bifrostErr.GetErrorString())
					}
					return resp.ExtraFields.Provider, take()
				}
				ch, bifrostErr := client.ChatCompletionStreamRequest(newPassthroughContext(t), chat(provider))
				if bifrostErr != nil {
					t.Fatalf("%s: %s", provider, bifrostErr.GetErrorString())
				}
				var named schemas.ModelProvider
				for chunk := range ch {
					if chunk.BifrostError != nil {
						t.Fatalf("%s stream: %s", provider, chunk.BifrostError.GetErrorString())
					}
					if chunk.BifrostChatResponse != nil {
						named = chunk.BifrostChatResponse.ExtraFields.Provider
					}
				}
				return named, take()
			}
			nativeName, nativeWire := send(schemas.Vertex)
			customName, customWire := send(customVertex)
			if nativeName != schemas.Vertex || customName != customVertex {
				t.Fatalf("responses name %q and %q, want vertex and %s", nativeName, customName, customVertex)
			}
			if len(nativeWire) != 1 || len(customWire) != 1 {
				t.Fatalf("upstream calls: native %d, custom %d, want 1 each", len(nativeWire), len(customWire))
			}
			if !strings.HasPrefix(customWire[0].path, wantPath) {
				t.Fatalf("custom path %q, want prefix %q", customWire[0].path, wantPath)
			}
			if customWire[0] != nativeWire[0] {
				t.Fatalf("custom wire request %+v differs from native %+v", customWire[0], nativeWire[0])
			}
		})
	}
}

// TestFileUploadContentCheckByBase pins that the empty-file exemption is Vertex's as a base (a
// Vertex upload may name a GCS source the provider reads itself), so a custom provider on vertex
// gets it and one on another base does not.
func TestFileUploadContentCheckByBase(t *testing.T) {
	client, _ := newVertexCustomBaseClient(t)
	const contentRequired = "file content is required"
	for _, tt := range []struct {
		provider schemas.ModelProvider
		required bool
	}{{schemas.Vertex, false}, {customVertex, false}, {schemas.OpenAI, true}} {
		t.Run(string(tt.provider), func(t *testing.T) {
			_, bifrostErr := client.FileUploadRequest(newPassthroughContext(t), &schemas.BifrostFileUploadRequest{Provider: tt.provider})
			got := bifrostErr != nil && bifrostErr.Error != nil && strings.HasPrefix(bifrostErr.Error.Message, contentRequired)
			if got != tt.required {
				t.Fatalf("content required = %v, want %v (err %+v)", got, tt.required, bifrostErr)
			}
		})
	}
}

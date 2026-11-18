package bifrost

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

// Validation refusals fire before the pipeline runs, so a zero-value Bifrost is enough to
// exercise them; each names the streamed-download request type.
func TestVideoDownloadStreamRequestValidation(t *testing.T) {
	bifrost := &Bifrost{}
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	tests := []struct {
		name string
		req  *schemas.BifrostVideoDownloadStreamRequest
		want string
	}{
		{"nil request", nil, "video download request is nil"},
		{"empty provider", &schemas.BifrostVideoDownloadStreamRequest{ID: "op"}, "provider is required"},
		{"empty id", &schemas.BifrostVideoDownloadStreamRequest{Provider: schemas.Vertex}, "video_id is required"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader, contentType, bifrostErr := bifrost.VideoDownloadStreamRequest(ctx, tt.req)
			if bifrostErr == nil {
				t.Fatal("validation refusal missing")
			}
			if reader != nil || contentType != "" {
				t.Fatalf("refusal returned reader=%v contentType=%q", reader, contentType)
			}
			if msg := bifrostErr.GetErrorString(); !strings.Contains(msg, tt.want) {
				t.Fatalf("error = %q, want it to contain %q", msg, tt.want)
			}
			if got := bifrostErr.ExtraFields.RequestType; got != schemas.VideoDownloadStreamRequest {
				t.Fatalf("request type = %q, want %q", got, schemas.VideoDownloadStreamRequest)
			}
		})
	}
}

// streamPipelineOperation is a Vertex video operation whose name carries a Veo model, the shape
// production IDs have.
const streamPipelineOperation = "projects/pipeline-proj/locations/us-central1/publishers/google/models/veo-3.0-generate-001/operations/op-9"

// newVideoStreamPipelineClient runs the real pipeline against a fake Vertex upstream: the
// retrieve lands on :fetchPredictOperation, every other request on artifact. The key is scoped
// to one Veo model — the normal production setup — not to "*".
func newVideoStreamPipelineClient(t *testing.T, artifact func(w http.ResponseWriter, r *http.Request), plugins ...schemas.LLMPlugin) *Bifrost {
	t.Helper()
	// The retrieve body points the download at this same server, so it is built inside the
	// handler where server.URL already exists.
	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, ":fetchPredictOperation") {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintf(w, `{"name":%q,"done":true,"response":{"generateVideoResponse":{"generatedSamples":[{"video":{"uri":%q,"mimeType":"video/mp4"}}]}}}`,
				streamPipelineOperation, server.URL+"/video.bin")
			return
		}
		artifact(w, r)
	}))
	t.Cleanup(server.Close)

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.Vertex, 1, 10, server.URL)
	account.configs[schemas.Vertex].NetworkConfig.AllowPrivateNetwork = true
	account.SetKeysForProvider(schemas.Vertex, []schemas.Key{{
		ID:     "veo-key",
		Value:  *schemas.NewSecretVar("api-key"),
		Models: schemas.WhiteList{"veo-3.0-generate-001"},
		Weight: 1,
		VertexKeyConfig: &schemas.VertexKeyConfig{
			ProjectID: *schemas.NewSecretVar("pipeline-proj"),
			Region:    *schemas.NewSecretVar("us-central1"),
		},
	}})
	return newStreamTestClient(t, account, plugins...)
}

// The streamed download carries no model, so key selection must skip the model filter for it:
// with a key scoped to Veo models (anything but ["*"]), the unlisted request type would run the
// full filter against model "" and the worker would 400 NoKeySupportsModel before the provider
// is ever reached. Proves the whole pipeline: Init, hooks, key selection, provider, stream back.
func TestVideoDownloadStream_FullPipelineWithModelScopedKey(t *testing.T) {
	const artifact = "pipeline-video-bytes"
	client := newVideoStreamPipelineClient(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "video/mp4")
		_, _ = w.Write([]byte(artifact))
	})

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	body, contentType, bifrostErr := client.VideoDownloadStreamRequest(ctx, &schemas.BifrostVideoDownloadStreamRequest{
		Provider: schemas.Vertex,
		ID:       streamPipelineOperation,
	})
	if bifrostErr != nil {
		t.Fatalf("VideoDownloadStreamRequest: %v", bifrostErr.GetErrorString())
	}
	defer body.Close()

	if contentType != "video/mp4" {
		t.Fatalf("content type = %q, want video/mp4", contentType)
	}
	got, err := io.ReadAll(body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	if string(got) != artifact {
		t.Fatalf("body = %q, want %q", got, artifact)
	}
}

// closeRecordingBody records Close so a test can prove core released a stream it dropped.
type closeRecordingBody struct {
	io.ReadCloser
	closed atomic.Bool
}

func (b *closeRecordingBody) Close() error {
	b.closed.Store(true)
	return b.ReadCloser.Close()
}

// streamInvalidatingPlugin wraps the stream body with a recorder, then invalidates the response
// in PostLLMHook the way a guardrail would: response dropped, error returned.
type streamInvalidatingPlugin struct {
	body *closeRecordingBody
}

func (p *streamInvalidatingPlugin) GetName() string { return "stream-invalidating" }
func (p *streamInvalidatingPlugin) Cleanup() error  { return nil }

func (p *streamInvalidatingPlugin) PreRequestHook(_ *schemas.BifrostContext, _ *schemas.BifrostRequest) error {
	return nil
}

func (p *streamInvalidatingPlugin) PreLLMHook(_ *schemas.BifrostContext, req *schemas.BifrostRequest) (*schemas.BifrostRequest, *schemas.LLMPluginShortCircuit, error) {
	return req, nil, nil
}

func (p *streamInvalidatingPlugin) PostLLMHook(_ *schemas.BifrostContext, resp *schemas.BifrostResponse, _ *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError, error) {
	if resp != nil && resp.VideoDownloadStreamResponse != nil && resp.VideoDownloadStreamResponse.Body != nil {
		p.body = &closeRecordingBody{ReadCloser: resp.VideoDownloadStreamResponse.Body}
		resp.VideoDownloadStreamResponse.Body = p.body
	}
	return nil, &schemas.BifrostError{
		IsBifrostError: false,
		Error:          &schemas.ErrorField{Message: "guardrail rejected the artifact"},
	}, nil
}

// A post hook that fails drops the response before the caller ever sees the reader; core must
// close the open stream body on that path or the connection leaks with nobody holding it.
func TestVideoDownloadStream_PostHookErrorClosesBody(t *testing.T) {
	plugin := &streamInvalidatingPlugin{}
	client := newVideoStreamPipelineClient(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "video/mp4")
		_, _ = w.Write([]byte("never-delivered"))
	}, plugin)

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	body, _, bifrostErr := client.VideoDownloadStreamRequest(ctx, &schemas.BifrostVideoDownloadStreamRequest{
		Provider: schemas.Vertex,
		ID:       streamPipelineOperation,
	})
	if bifrostErr == nil {
		body.Close()
		t.Fatal("invalidating post hook did not fail the request")
	}
	if body != nil {
		t.Fatal("errored request handed the caller a reader")
	}
	if plugin.body == nil {
		t.Fatal("post hook never saw the stream body")
	}
	if !plugin.body.closed.Load() {
		t.Fatal("stream body left open when the post-hook error dropped the response")
	}
}

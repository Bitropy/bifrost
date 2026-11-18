package gemini

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
)

const geminiStreamOperation = "models/veo-3.0-generate-preview/operations/op-1"

func geminiRetrievePath() string { return "/" + geminiStreamOperation }

// geminiCompletedOperation is the operation-retrieve body for a done job whose artifact lives
// at downloadURI; geminiInlineOperation is one whose artifact came back inline.
func geminiCompletedOperation(downloadURI string) string {
	return fmt.Sprintf(`{"name":%q,"done":true,"response":{"generateVideoResponse":{"generatedSamples":[{"video":{"uri":%q,"mimeType":"video/mp4"}}]}}}`,
		geminiStreamOperation, downloadURI)
}

var geminiInlineOperation = fmt.Sprintf(`{"name":%q,"done":true,"response":{"generateVideoResponse":{"generatedSamples":[{"video":{"videoBytes":%q,"mimeType":"video/mp4"}}]}}}`,
	geminiStreamOperation, "aW5saW5lLXZpZGVv")

// startGeminiVideoStreamServer serves the operation retrieve at the operation path and hands
// every other request to artifact. The retrieve body is written through the *string so a test
// can point the artifact URL at this server once its URL exists.
func startGeminiVideoStreamServer(t *testing.T, retrieve *string, artifact func(w http.ResponseWriter, r *http.Request)) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == geminiRetrievePath() {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(*retrieve))
			return
		}
		artifact(w, r)
	}))
	t.Cleanup(server.Close)
	return server
}

func newGeminiStreamProvider(t *testing.T, serverURL string) *GeminiProvider {
	t.Helper()
	provider := NewGeminiProvider(&schemas.ProviderConfig{
		NetworkConfig: schemas.NetworkConfig{BaseURL: serverURL, DefaultRequestTimeoutInSeconds: 30},
	}, testNoopLogger{})
	t.Cleanup(provider.client.CloseIdleConnections)
	t.Cleanup(provider.videoDownloadClient.CloseIdleConnections)
	return provider
}

func geminiStreamRequest() *schemas.BifrostVideoDownloadStreamRequest {
	return &schemas.BifrostVideoDownloadStreamRequest{Provider: schemas.Gemini, ID: geminiStreamOperation}
}

func geminiStreamKey() schemas.Key {
	return schemas.Key{Value: *schemas.NewSecretVar("gemini-test")}
}

// The artifact URL redirects (signed-URL hop): the stream must follow it, keep the API key
// header on the download, and pass the artifact's content type through.
func TestGeminiVideoDownloadStreamFollowsRedirect(t *testing.T) {
	const artifact = "tiny-video-bytes"
	var gotAPIKey string
	var artifactHits int
	retrieve := "unset"
	redirectTarget := "unset"
	server := startGeminiVideoStreamServer(t, &retrieve, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/redirect":
			w.Header().Set("Location", redirectTarget)
			w.WriteHeader(http.StatusFound)
		case "/artifact":
			artifactHits++
			gotAPIKey = r.Header.Get("x-goog-api-key")
			w.Header().Set("Content-Type", "video/webm")
			_, _ = w.Write([]byte(artifact))
		default:
			t.Errorf("unexpected artifact path %q", r.URL.Path)
		}
	})
	retrieve = geminiCompletedOperation(server.URL + "/redirect")
	redirectTarget = server.URL + "/artifact"

	provider := newGeminiStreamProvider(t, server.URL)
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	resp, bifrostErr := provider.VideoDownloadStream(ctx, geminiStreamKey(), geminiStreamRequest())
	if bifrostErr != nil {
		t.Fatalf("VideoDownloadStream: %v", bifrostErr.GetErrorString())
	}
	defer resp.Body.Close()

	if resp.ContentType != "video/webm" {
		t.Fatalf("content type = %q, want video/webm", resp.ContentType)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	if string(body) != artifact {
		t.Fatalf("body = %q, want %q", body, artifact)
	}
	if artifactHits != 1 {
		t.Fatalf("artifact hits = %d, want 1", artifactHits)
	}
	if gotAPIKey != "gemini-test" {
		t.Fatalf("download x-goog-api-key = %q, want the key credential", gotAPIKey)
	}
}

// A 64 MiB artifact must stream through without the provider ever holding it: heap is sampled
// while reading and has to stay within 4 MiB of the pre-request baseline.
func TestGeminiVideoDownloadStreamDoesNotBuffer64MiB(t *testing.T) {
	const (
		totalMiB  = 64
		heapCeil  = 4 << 20
		chunkSize = 64 << 10
	)
	block := make([]byte, 1<<20)
	for i := range block {
		block[i] = byte(i)
	}
	retrieve := "unset"
	server := startGeminiVideoStreamServer(t, &retrieve, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "video/mp4")
		w.Header().Set("Content-Length", fmt.Sprintf("%d", totalMiB<<20))
		for i := 0; i < totalMiB; i++ {
			if _, err := w.Write(block); err != nil {
				return
			}
		}
	})
	retrieve = geminiCompletedOperation(server.URL + "/video.bin")

	provider := newGeminiStreamProvider(t, server.URL)
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	runtime.GC()
	var base runtime.MemStats
	runtime.ReadMemStats(&base)

	resp, bifrostErr := provider.VideoDownloadStream(ctx, geminiStreamKey(), geminiStreamRequest())
	if bifrostErr != nil {
		t.Fatalf("VideoDownloadStream: %v", bifrostErr.GetErrorString())
	}
	defer resp.Body.Close()

	buf := make([]byte, chunkSize)
	var total int
	for {
		n, err := resp.Body.Read(buf)
		total += n
		if total%(8<<20) < chunkSize {
			var ms runtime.MemStats
			runtime.ReadMemStats(&ms)
			if grew := int64(ms.HeapAlloc) - int64(base.HeapAlloc); grew > heapCeil {
				t.Fatalf("heap grew by %d MiB after %d MiB read; the artifact is being buffered", grew>>20, total>>20)
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("read at %d MiB: %v", total>>20, err)
		}
	}
	if total != totalMiB<<20 {
		t.Fatalf("read %d bytes, want %d", total, totalMiB<<20)
	}
}

// Cancelling the caller's context must close the stream: a blocked read returns an error
// instead of waiting out the idle timeout.
func TestGeminiVideoDownloadStreamCancelClosesReader(t *testing.T) {
	release := make(chan struct{})
	retrieve := "unset"
	server := startGeminiVideoStreamServer(t, &retrieve, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "video/mp4")
		w.Header().Set("Content-Length", fmt.Sprintf("%d", 1<<20))
		_, _ = w.Write(make([]byte, 64<<10))
		<-release
	})
	t.Cleanup(func() { close(release) })
	retrieve = geminiCompletedOperation(server.URL + "/video.bin")

	provider := newGeminiStreamProvider(t, server.URL)
	goCtx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	ctx := schemas.NewBifrostContext(goCtx, schemas.NoDeadline)

	resp, bifrostErr := provider.VideoDownloadStream(ctx, geminiStreamKey(), geminiStreamRequest())
	if bifrostErr != nil {
		t.Fatalf("VideoDownloadStream: %v", bifrostErr.GetErrorString())
	}
	defer resp.Body.Close()

	// Drain the bytes the server already wrote so the next read is the blocked one.
	if _, err := io.ReadFull(resp.Body, make([]byte, 64<<10)); err != nil {
		t.Fatalf("initial read: %v", err)
	}
	cancel()

	outcome := make(chan error, 1)
	go func() {
		_, err := resp.Body.Read(make([]byte, 64<<10))
		outcome <- err
	}()
	select {
	case err := <-outcome:
		if err == nil {
			t.Fatal("read after cancel succeeded; the stream must close on cancellation")
		}
	case <-time.After(10 * time.Second):
		t.Fatal("read after cancel did not return within 10s; the stream does not close on cancellation")
	}
}

// An artifact the provider only holds inline (base64 in the retrieve response) is refused with
// a clear error naming the buffered download, never silently materialized.
func TestGeminiVideoDownloadStreamRefusesInlineArtifact(t *testing.T) {
	retrieve := geminiInlineOperation
	server := startGeminiVideoStreamServer(t, &retrieve, func(w http.ResponseWriter, _ *http.Request) {
		t.Error("artifact endpoint reached for an inline-only operation")
	})

	provider := newGeminiStreamProvider(t, server.URL)
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	_, bifrostErr := provider.VideoDownloadStream(ctx, geminiStreamKey(), geminiStreamRequest())
	if bifrostErr == nil {
		t.Fatal("inline artifact accepted by the streamed download")
	}
	if msg := bifrostErr.GetErrorString(); !strings.Contains(msg, "inline") || !strings.Contains(msg, "buffered") {
		t.Fatalf("error = %q, want it to name the inline artifact and the buffered download", msg)
	}
}

// A job that is not completed yet is refused exactly like the buffered download.
func TestGeminiVideoDownloadStreamNotReady(t *testing.T) {
	retrieve := fmt.Sprintf(`{"name":%q,"done":false}`, geminiStreamOperation)
	server := startGeminiVideoStreamServer(t, &retrieve, func(w http.ResponseWriter, _ *http.Request) {
		t.Error("artifact endpoint reached for an in-progress operation")
	})

	provider := newGeminiStreamProvider(t, server.URL)
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	_, bifrostErr := provider.VideoDownloadStream(ctx, geminiStreamKey(), geminiStreamRequest())
	if bifrostErr == nil {
		t.Fatal("download of an in-progress job succeeded")
	}
	if msg := bifrostErr.GetErrorString(); !strings.Contains(msg, "video not ready") {
		t.Fatalf("error = %q, want video not ready", msg)
	}
}

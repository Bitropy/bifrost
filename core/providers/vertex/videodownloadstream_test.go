package vertex

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
)

const streamTestOperation = "projects/proj/locations/us-central1/publishers/google/models/veo-3.0-generate-001/operations/op-1"

func newVideoStreamKey() schemas.Key {
	return schemas.Key{
		Value:           *schemas.NewSecretVar("api-key"),
		VertexKeyConfig: &schemas.VertexKeyConfig{ProjectID: *schemas.NewSecretVar("proj"), Region: *schemas.NewSecretVar("us-central1")},
	}
}

func newVideoStreamProvider(t *testing.T, serverURL string) *VertexProvider {
	t.Helper()
	config := &schemas.ProviderConfig{
		NetworkConfig: schemas.NetworkConfig{DefaultRequestTimeoutInSeconds: 30, BaseURL: serverURL, AllowPrivateNetwork: true},
	}
	config.CheckAndSetDefaults()
	provider, err := NewVertexProvider(config, nil)
	if err != nil {
		t.Fatalf("NewVertexProvider: %v", err)
	}
	t.Cleanup(provider.client.CloseIdleConnections)
	t.Cleanup(provider.videoDownloadClient.CloseIdleConnections)
	return provider
}

// completedOperationJSON is the fetchPredictOperation body for a done job whose artifact lives
// at downloadURI; inlineOperationJSON is one whose artifact came back as base64.
func completedOperationJSON(operation, downloadURI string) string {
	return fmt.Sprintf(`{"name":%q,"done":true,"response":{"generateVideoResponse":{"generatedSamples":[{"video":{"uri":%q,"mimeType":"video/mp4"}}]}}}`,
		operation, downloadURI)
}

func inlineOperationJSON(operation string) string {
	return fmt.Sprintf(`{"name":%q,"done":true,"response":{"generateVideoResponse":{"generatedSamples":[{"video":{"videoBytes":%q,"mimeType":"video/mp4"}}]}}}`,
		operation, "aW5saW5lLXZpZGVv")
}

// startVideoStreamServer serves the retrieve operation at :fetchPredictOperation and hands every
// other request to artifact. The retrieve body is written through the *string so a test can point
// the artifact URL at this server once its URL exists.
func startVideoStreamServer(t *testing.T, retrieve *string, artifact func(w http.ResponseWriter, r *http.Request)) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, ":fetchPredictOperation") {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(*retrieve))
			return
		}
		artifact(w, r)
	}))
	t.Cleanup(server.Close)
	return server
}

func streamRequest(operation string) *schemas.BifrostVideoDownloadStreamRequest {
	return &schemas.BifrostVideoDownloadStreamRequest{Provider: schemas.Vertex, ID: operation}
}

// The happy path hands back the artifact URL's body unbuffered with its content type, and the
// download GET carries the key's credential in the query string like the buffered download.
func TestVertexVideoDownloadStreamSmallBody(t *testing.T) {
	const artifact = "tiny-video-bytes"
	var gotQuery string
	retrieve := "unset"
	server := startVideoStreamServer(t, &retrieve, func(w http.ResponseWriter, r *http.Request) {
		if gotQuery == "" {
			gotQuery = r.URL.RawQuery
		}
		w.Header().Set("Content-Type", "video/mp4")
		_, _ = w.Write([]byte(artifact))
	})
	retrieve = completedOperationJSON(streamTestOperation, server.URL+"/video.bin")

	provider := newVideoStreamProvider(t, server.URL)
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	resp, bifrostErr := provider.VideoDownloadStream(ctx, newVideoStreamKey(), streamRequest(streamTestOperation))
	if bifrostErr != nil {
		t.Fatalf("VideoDownloadStream: %v", bifrostErr.GetErrorString())
	}
	defer resp.Body.Close()

	if resp.ContentType != "video/mp4" {
		t.Fatalf("content type = %q, want video/mp4", resp.ContentType)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	if string(body) != artifact {
		t.Fatalf("body = %q, want %q", body, artifact)
	}
	if !strings.Contains(gotQuery, "key=api-key") {
		t.Fatalf("download query = %q, want the API key credential", gotQuery)
	}
}

// A 64 MiB artifact must stream through without the provider ever holding it: heap is sampled
// while reading and has to stay within 4 MiB of the pre-request baseline.
func TestVertexVideoDownloadStreamDoesNotBuffer64MiB(t *testing.T) {
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
	server := startVideoStreamServer(t, &retrieve, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "video/mp4")
		w.Header().Set("Content-Length", fmt.Sprintf("%d", totalMiB<<20))
		for i := 0; i < totalMiB; i++ {
			if _, err := w.Write(block); err != nil {
				return
			}
		}
	})
	retrieve = completedOperationJSON(streamTestOperation, server.URL+"/video.bin")

	provider := newVideoStreamProvider(t, server.URL)
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	runtime.GC()
	var base runtime.MemStats
	runtime.ReadMemStats(&base)

	resp, bifrostErr := provider.VideoDownloadStream(ctx, newVideoStreamKey(), streamRequest(streamTestOperation))
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
func TestVertexVideoDownloadStreamCancelClosesReader(t *testing.T) {
	release := make(chan struct{})
	retrieve := "unset"
	server := startVideoStreamServer(t, &retrieve, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "video/mp4")
		w.Header().Set("Content-Length", fmt.Sprintf("%d", 1<<20))
		_, _ = w.Write(make([]byte, 64<<10))
		<-release
	})
	t.Cleanup(func() { close(release) })
	retrieve = completedOperationJSON(streamTestOperation, server.URL+"/video.bin")

	provider := newVideoStreamProvider(t, server.URL)
	goCtx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	ctx := schemas.NewBifrostContext(goCtx, schemas.NoDeadline)

	resp, bifrostErr := provider.VideoDownloadStream(ctx, newVideoStreamKey(), streamRequest(streamTestOperation))
	if bifrostErr != nil {
		t.Fatalf("VideoDownloadStream: %v", bifrostErr.GetErrorString())
	}
	defer resp.Body.Close()

	// Drain the bytes the server already wrote so the next read is the blocked one.
	if _, err := io.ReadFull(resp.Body, make([]byte, 64<<10)); err != nil {
		t.Fatalf("initial read: %v", err)
	}
	cancel()

	type readOutcome struct {
		err error
	}
	outcome := make(chan readOutcome, 1)
	go func() {
		_, err := resp.Body.Read(make([]byte, 64<<10))
		outcome <- readOutcome{err}
	}()
	select {
	case got := <-outcome:
		if got.err == nil {
			t.Fatal("read after cancel succeeded; the stream must close on cancellation")
		}
	case <-time.After(10 * time.Second):
		t.Fatal("read after cancel did not return within 10s; the stream does not close on cancellation")
	}
}

// An artifact the provider only holds inline (base64 in the retrieve response) is refused with
// a clear error naming the buffered download, never silently materialized.
func TestVertexVideoDownloadStreamRefusesInlineArtifact(t *testing.T) {
	retrieve := inlineOperationJSON(streamTestOperation)
	server := startVideoStreamServer(t, &retrieve, func(w http.ResponseWriter, _ *http.Request) {
		t.Error("artifact endpoint reached for an inline-only operation")
	})

	provider := newVideoStreamProvider(t, server.URL)
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	_, bifrostErr := provider.VideoDownloadStream(ctx, newVideoStreamKey(), streamRequest(streamTestOperation))
	if bifrostErr == nil {
		t.Fatal("inline artifact accepted by the streamed download")
	}
	if msg := bifrostErr.GetErrorString(); !strings.Contains(msg, "inline") || !strings.Contains(msg, "buffered") {
		t.Fatalf("error = %q, want it to name the inline artifact and the buffered download", msg)
	}
}

// A job that is not completed yet is refused exactly like the buffered download.
func TestVertexVideoDownloadStreamNotReady(t *testing.T) {
	retrieve := fmt.Sprintf(`{"name":%q,"done":false}`, streamTestOperation)
	server := startVideoStreamServer(t, &retrieve, func(w http.ResponseWriter, _ *http.Request) {
		t.Error("artifact endpoint reached for an in-progress operation")
	})

	provider := newVideoStreamProvider(t, server.URL)
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	_, bifrostErr := provider.VideoDownloadStream(ctx, newVideoStreamKey(), streamRequest(streamTestOperation))
	if bifrostErr == nil {
		t.Fatal("download of an in-progress job succeeded")
	}
	if msg := bifrostErr.GetErrorString(); !strings.Contains(msg, "video not ready") {
		t.Fatalf("error = %q, want video not ready", msg)
	}
}

// A body that runs past MaxVideoDownloadStreamBytes errors at exactly the cap instead of
// streaming forever, and the capped read never buffers the artifact: heap is sampled while
// reading and has to stay within 4 MiB of the pre-request baseline, as in the 64 MiB proof.
func TestVertexVideoDownloadStreamCapExceeded(t *testing.T) {
	const (
		chunkSize = 64 << 10
		heapCeil  = 4 << 20
	)
	block := make([]byte, 1<<20)
	retrieve := "unset"
	server := startVideoStreamServer(t, &retrieve, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "video/mp4")
		for {
			if _, err := w.Write(block); err != nil {
				return
			}
		}
	})
	retrieve = completedOperationJSON(streamTestOperation, server.URL+"/video.bin")

	provider := newVideoStreamProvider(t, server.URL)
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	runtime.GC()
	var base runtime.MemStats
	runtime.ReadMemStats(&base)

	resp, bifrostErr := provider.VideoDownloadStream(ctx, newVideoStreamKey(), streamRequest(streamTestOperation))
	if bifrostErr != nil {
		t.Fatalf("VideoDownloadStream: %v", bifrostErr.GetErrorString())
	}
	defer resp.Body.Close()

	buf := make([]byte, chunkSize)
	var total int64
	for {
		n, err := resp.Body.Read(buf)
		total += int64(n)
		if total%(32<<20) < chunkSize {
			var ms runtime.MemStats
			runtime.ReadMemStats(&ms)
			if grew := int64(ms.HeapAlloc) - int64(base.HeapAlloc); grew > heapCeil {
				t.Fatalf("heap grew by %d MiB after %d MiB read; the artifact is being buffered", grew>>20, total>>20)
			}
		}
		if err != nil {
			if !errors.Is(err, utils.ErrVideoDownloadStreamCapExceeded) {
				t.Fatalf("read at %d MiB: %v, want the byte-cap error", total>>20, err)
			}
			break
		}
	}
	if total != utils.MaxVideoDownloadStreamBytes {
		t.Fatalf("read %d bytes before the cap error, want exactly the cap (%d)", total, utils.MaxVideoDownloadStreamBytes)
	}
}

// The drain ReleaseStreamingResponse runs for a non-200 response is bounded at the same cap:
// a 500 whose body never ends must still produce the HTTP error instead of blocking forever
// on an unbounded drain.
func TestVertexVideoDownloadStreamNon200DrainIsCapped(t *testing.T) {
	block := make([]byte, 1<<20)
	retrieve := "unset"
	server := startVideoStreamServer(t, &retrieve, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		for {
			if _, err := w.Write(block); err != nil {
				return
			}
		}
	})
	retrieve = completedOperationJSON(streamTestOperation, server.URL+"/video.bin")

	provider := newVideoStreamProvider(t, server.URL)
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	// Empty string means the download failed with the HTTP 500 error as it should.
	result := make(chan string, 1)
	go func() {
		_, bifrostErr := provider.VideoDownloadStream(ctx, newVideoStreamKey(), streamRequest(streamTestOperation))
		if bifrostErr == nil {
			result <- "download of an endless 500 body succeeded"
			return
		}
		if msg := bifrostErr.GetErrorString(); !strings.Contains(msg, "HTTP 500") {
			result <- fmt.Sprintf("error = %q, want HTTP 500", msg)
			return
		}
		result <- ""
	}()
	select {
	case failure := <-result:
		if failure != "" {
			t.Fatal(failure)
		}
	case <-time.After(2 * time.Minute):
		t.Fatal("non-200 download did not return within 2m; the release drain is unbounded")
	}
}

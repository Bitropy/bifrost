package bifrost

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
)

// The providers that serve batch files and batches (BIT-2923, BIT-2924), each with the
// passthrough path a batch upload takes.
var streamedPassthroughProviders = []struct {
	provider schemas.ModelProvider
	path     string
}{
	{schemas.OpenAI, "/v1/files"},
	{schemas.Azure, "/openai/v1/files"},
	{schemas.Anthropic, "/v1/messages/batches"},
}

func newStreamedPassthroughClient(t *testing.T, provider schemas.ModelProvider, url string, maxRetries int) *Bifrost {
	t.Helper()
	account := NewMockAccount()
	configureUpstream(account, provider, "key-"+string(provider), url, maxRetries)
	client, err := Init(context.Background(), schemas.BifrostConfig{Account: account, Logger: NewDefaultLogger(schemas.LogLevelError)})
	if err != nil {
		t.Fatalf("Init: %v", err)
	}
	t.Cleanup(client.Shutdown)
	return client
}

func newPassthroughContext(t *testing.T) *schemas.BifrostContext {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	return schemas.NewBifrostContext(ctx, schemas.NoDeadline)
}

// F1: BodyReader goes out as it is read, chunked when its size is unknown and with a
// Content-Length when it is known, byte-exact either way.
func TestStreamedPassthroughRequestBody(t *testing.T) {
	payload := bytes.Repeat([]byte(`{"custom_id":"r","body":{"model":"m"}}`+"\n"), 4096)
	for _, p := range streamedPassthroughProviders {
		for _, sized := range []bool{false, true} {
			name := string(p.provider) + "/chunked"
			if sized {
				name = string(p.provider) + "/sized"
			}
			t.Run(name, func(t *testing.T) {
				var gotBody []byte
				var gotTE []string
				var gotLen int64
				srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					gotTE, gotLen = r.TransferEncoding, r.ContentLength
					gotBody, _ = io.ReadAll(r.Body)
					w.Header().Set("Content-Type", "application/json")
					_, _ = io.WriteString(w, `{"id":"file-1"}`)
				}))
				t.Cleanup(srv.Close)
				client := newStreamedPassthroughClient(t, p.provider, srv.URL, 0)

				size := int64(-1)
				if sized {
					size = int64(len(payload))
				}
				resp, bifrostErr := client.Passthrough(newPassthroughContext(t), p.provider, &schemas.BifrostPassthroughRequest{
					Method:      http.MethodPost,
					Path:        p.path,
					SafeHeaders: map[string]string{"Content-Type": "application/jsonl"},
					BodyReader:  bytes.NewReader(payload),
					BodySize:    size,
				})
				if bifrostErr != nil {
					t.Fatalf("Passthrough: %+v", bifrostErr.Error)
				}
				if resp.StatusCode != http.StatusOK || string(resp.Body) != `{"id":"file-1"}` {
					t.Fatalf("response = %d %q", resp.StatusCode, resp.Body)
				}
				if !bytes.Equal(gotBody, payload) {
					t.Fatalf("upstream got %d bytes, want the %d sent", len(gotBody), len(payload))
				}
				if sized {
					if gotLen != int64(len(payload)) || len(gotTE) != 0 {
						t.Fatalf("sized body: Content-Length %d, Transfer-Encoding %v", gotLen, gotTE)
					}
				} else if len(gotTE) != 1 || gotTE[0] != "chunked" {
					t.Fatalf("unsized body: Transfer-Encoding %v, want chunked", gotTE)
				}
			})
		}
	}
}

// F1: a BodyReader can be read once, so a request carrying one gets exactly one attempt even
// when the failure is one core retries. A re-send would go out with the drained reader, an
// empty body, and the upstream would act on it.
func TestStreamedPassthroughRequestIsSentOnce(t *testing.T) {
	for _, p := range streamedPassthroughProviders {
		t.Run(string(p.provider), func(t *testing.T) {
			var hits atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				hits.Add(1)
				_, _ = io.Copy(io.Discard, r.Body)
				// A dropped connection is a transient upstream failure core would retry.
				conn, _, err := w.(http.Hijacker).Hijack()
				if err == nil {
					_ = conn.Close()
				}
			}))
			t.Cleanup(srv.Close)

			// Control: the same failure with a byte Body is retried, so the case is not vacuous.
			control := newStreamedPassthroughClient(t, p.provider, srv.URL, 2)
			_, controlErr := control.Passthrough(newPassthroughContext(t), p.provider, &schemas.BifrostPassthroughRequest{
				Method: http.MethodPost, Path: p.path, Body: []byte("x"),
			})
			if controlErr == nil || hits.Load() != 3 {
				t.Fatalf("control: %d upstream hits (err %v), want 3", hits.Load(), controlErr)
			}

			hits.Store(0)
			client := newStreamedPassthroughClient(t, p.provider, srv.URL, 2)
			ctx := newPassthroughContext(t)
			_, bifrostErr := client.Passthrough(ctx, p.provider, &schemas.BifrostPassthroughRequest{
				Method: http.MethodPost, Path: p.path, BodyReader: strings.NewReader("x"), BodySize: -1,
			})
			if bifrostErr == nil {
				t.Fatal("want the dropped connection's error")
			}
			if n := hits.Load(); n != 1 {
				t.Fatalf("upstream hits = %d, want 1", n)
			}
			if trail, _ := ctx.Value(schemas.BifrostContextKeyAttemptTrail).([]schemas.KeyAttemptRecord); len(trail) != 1 {
				t.Fatalf("attempt trail has %d record(s), want 1", len(trail))
			}
		})
	}
}

// failingReader yields its data, then a non-EOF error, like a batch body whose line N
// fails validation after lines 1..N-1 were sent.
type failingReader struct {
	data []byte
	err  error
}

func (r *failingReader) Read(p []byte) (int, error) {
	if len(r.data) == 0 {
		return 0, r.err
	}
	n := copy(p, r.data)
	r.data = r.data[n:]
	return n, nil
}

// rawUpstream accepts connections and records every byte each one carries, so a test can
// see the request exactly as the wire framed it.
type rawUpstream struct {
	ln       net.Listener
	mu       sync.Mutex
	conns    [][]byte
	recorded chan struct{}
}

func newRawUpstream(t *testing.T) *rawUpstream {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	u := &rawUpstream{ln: ln, recorded: make(chan struct{}, 16)}
	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go func() {
				defer conn.Close()
				_ = conn.SetReadDeadline(time.Now().Add(10 * time.Second))
				data, _ := io.ReadAll(conn)
				u.mu.Lock()
				u.conns = append(u.conns, data)
				u.mu.Unlock()
				u.recorded <- struct{}{}
			}()
		}
	}()
	t.Cleanup(func() { _ = ln.Close() })
	return u
}

func (u *rawUpstream) url() string { return "http://" + u.ln.Addr().String() }

// received waits for the first connection to end and returns every connection recorded so far.
func (u *rawUpstream) received(t *testing.T) [][]byte {
	t.Helper()
	select {
	case <-u.recorded:
	case <-time.After(10 * time.Second):
		t.Fatal("no connection reached the upstream")
	}
	u.mu.Lock()
	defer u.mu.Unlock()
	return u.conns
}

// F1: a BodyReader that fails aborts the upstream request without its terminating chunk, so
// the upstream holds an incomplete request and creates nothing.
func TestStreamedPassthroughBodyAbortSendsNoTerminalChunk(t *testing.T) {
	readErr := errors.New("line 3: model not allowed")
	for _, p := range streamedPassthroughProviders {
		t.Run(string(p.provider), func(t *testing.T) {
			upstream := newRawUpstream(t)
			client := newStreamedPassthroughClient(t, p.provider, upstream.url(), 2)

			ctx := newPassthroughContext(t)
			_, bifrostErr := client.Passthrough(ctx, p.provider, &schemas.BifrostPassthroughRequest{
				Method:     http.MethodPost,
				Path:       p.path,
				BodyReader: &failingReader{data: []byte("line-1\nline-2\n"), err: readErr},
				BodySize:   -1,
			})
			if bifrostErr == nil {
				t.Fatal("want an error from the failed body")
			}
			if trail, _ := ctx.Value(schemas.BifrostContextKeyAttemptTrail).([]schemas.KeyAttemptRecord); len(trail) != 1 {
				t.Fatalf("attempt trail has %d record(s), want 1", len(trail))
			}

			conns := upstream.received(t)
			if len(conns) != 1 {
				t.Fatalf("upstream saw %d connections, want 1", len(conns))
			}
			wire := conns[0]
			headerEnd := bytes.Index(wire, []byte("\r\n\r\n"))
			if headerEnd < 0 || !bytes.Contains(bytes.ToLower(wire[:headerEnd]), []byte("transfer-encoding: chunked")) {
				t.Fatalf("request is not chunked: %q", wire)
			}
			body := wire[headerEnd+4:]
			if !bytes.Contains(body, []byte("line-1\nline-2\n")) {
				t.Fatalf("the bytes read before the failure were not sent: %q", body)
			}
			if bytes.Contains(body, []byte("0\r\n\r\n")) {
				t.Fatalf("upstream got a terminating chunk: %q", body)
			}
		})
	}
}

// F2: StreamResponseBody hands back status and headers before the body is read, the body
// itself as BodyStream, and nothing in Body.
func TestStreamedPassthroughResponseBody(t *testing.T) {
	first := bytes.Repeat([]byte(`{"custom_id":"a","response":{}}`+"\n"), 64)
	second := bytes.Repeat([]byte(`{"custom_id":"b","response":{}}`+"\n"), 64)
	for _, p := range streamedPassthroughProviders {
		t.Run(string(p.provider), func(t *testing.T) {
			release := make(chan struct{})
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/octet-stream")
				w.Header().Set("X-Upstream", "yes")
				_, _ = w.Write(first)
				w.(http.Flusher).Flush()
				// The rest waits for the test: Passthrough returning at all proves it did not
				// read the body first.
				select {
				case <-release:
				case <-r.Context().Done():
					return
				}
				_, _ = w.Write(second)
			}))
			t.Cleanup(srv.Close)
			client := newStreamedPassthroughClient(t, p.provider, srv.URL, 0)

			resp, bifrostErr := client.Passthrough(newPassthroughContext(t), p.provider, &schemas.BifrostPassthroughRequest{
				Method: http.MethodGet, Path: p.path + "/file-1/content", StreamResponseBody: true,
			})
			if bifrostErr != nil {
				t.Fatalf("Passthrough: %+v", bifrostErr.Error)
			}
			if resp.StatusCode != http.StatusOK || resp.BodyStream == nil || resp.Body != nil {
				t.Fatalf("status %d, BodyStream %v, Body %d bytes", resp.StatusCode, resp.BodyStream != nil, len(resp.Body))
			}
			if resp.Headers["X-Upstream"] != "yes" {
				t.Fatalf("headers = %v", resp.Headers)
			}
			got := make([]byte, len(first))
			if _, err := io.ReadFull(resp.BodyStream, got); err != nil || !bytes.Equal(got, first) {
				t.Fatalf("first part: %v", err)
			}
			close(release)
			rest, err := io.ReadAll(resp.BodyStream)
			if err != nil || !bytes.Equal(rest, second) {
				t.Fatalf("rest: %d bytes, %v", len(rest), err)
			}
			if err := resp.BodyStream.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// F2: a gzip body is decoded, and the returned headers no longer claim the encoding.
func TestStreamedPassthroughResponseBodyGzip(t *testing.T) {
	payload := bytes.Repeat([]byte(`{"custom_id":"a"}`+"\n"), 1000)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Encoding", "gzip")
		gz := gzip.NewWriter(w)
		_, _ = gz.Write(payload)
		_ = gz.Close()
	}))
	t.Cleanup(srv.Close)
	client := newStreamedPassthroughClient(t, schemas.OpenAI, srv.URL, 0)

	resp, bifrostErr := client.Passthrough(newPassthroughContext(t), schemas.OpenAI, &schemas.BifrostPassthroughRequest{
		Method: http.MethodGet, Path: "/v1/files/file-1/content", StreamResponseBody: true,
	})
	if bifrostErr != nil {
		t.Fatalf("Passthrough: %+v", bifrostErr.Error)
	}
	t.Cleanup(func() { _ = resp.BodyStream.Close() })
	got, err := io.ReadAll(resp.BodyStream)
	if err != nil || !bytes.Equal(got, payload) {
		t.Fatalf("decoded body: %d bytes, %v", len(got), err)
	}
	for k := range resp.Headers {
		if strings.EqualFold(k, "Content-Encoding") {
			t.Fatalf("headers still carry %s", k)
		}
	}
}

// F2: a non-2xx answer comes back in Body as today, read up to the bound, with no BodyStream. A
// body past the bound is abandoned, not drained, so one that never ends cannot hold the call.
func TestStreamedPassthroughResponseErrorBodyIsBounded(t *testing.T) {
	chunk := bytes.Repeat([]byte("e"), 16*1024)
	for _, p := range streamedPassthroughProviders {
		t.Run(string(p.provider), func(t *testing.T) {
			handlerDone, stop := make(chan struct{}), make(chan struct{})
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				defer close(handlerDone)
				w.WriteHeader(http.StatusNotFound)
				// Endless error body: only a closed connection (or a failed test) ends this handler.
				for {
					select {
					case <-stop:
						return
					default:
					}
					if _, err := w.Write(chunk); err != nil {
						return
					}
					w.(http.Flusher).Flush()
				}
			}))
			t.Cleanup(srv.Close)
			t.Cleanup(func() { close(stop) }) // runs first, so a draining client cannot hang srv.Close
			client := newStreamedPassthroughClient(t, p.provider, srv.URL, 0)

			type result struct {
				resp *schemas.BifrostPassthroughResponse
				err  *schemas.BifrostError
			}
			done := make(chan result, 1)
			go func() {
				resp, bifrostErr := client.Passthrough(newPassthroughContext(t), p.provider, &schemas.BifrostPassthroughRequest{
					Method: http.MethodGet, Path: p.path + "/file-1/content", StreamResponseBody: true,
				})
				done <- result{resp, bifrostErr}
			}()
			var got result
			select {
			case got = <-done:
			case <-time.After(10 * time.Second):
				t.Fatal("Passthrough drained an endless error body instead of abandoning it")
			}
			if got.err != nil {
				t.Fatalf("Passthrough: %+v", got.err.Error)
			}
			if got.resp.StatusCode != http.StatusNotFound || got.resp.BodyStream != nil {
				t.Fatalf("status %d, BodyStream %v", got.resp.StatusCode, got.resp.BodyStream != nil)
			}
			if len(got.resp.Body) != providerUtils.PassthroughErrorBodyLimit || got.resp.Body[0] != 'e' {
				t.Fatalf("Body has %d bytes, want the first %d", len(got.resp.Body), providerUtils.PassthroughErrorBodyLimit)
			}
			select {
			case <-handlerDone:
			case <-time.After(10 * time.Second):
				t.Fatal("the upstream connection was not closed")
			}
		})
	}
}

// F2: closing BodyStream before EOF abandons the connection instead of draining the rest of a
// body that can be hundreds of megabytes.
func TestStreamedPassthroughResponseEarlyCloseDoesNotDrain(t *testing.T) {
	chunk := bytes.Repeat([]byte("x"), 32*1024)
	for _, p := range streamedPassthroughProviders {
		t.Run(string(p.provider), func(t *testing.T) {
			handlerDone := make(chan struct{})
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				defer close(handlerDone)
				// Endless body: only a closed connection ends this handler.
				for {
					if _, err := w.Write(chunk); err != nil {
						return
					}
					w.(http.Flusher).Flush()
				}
			}))
			t.Cleanup(srv.Close)
			client := newStreamedPassthroughClient(t, p.provider, srv.URL, 0)

			resp, bifrostErr := client.Passthrough(newPassthroughContext(t), p.provider, &schemas.BifrostPassthroughRequest{
				Method: http.MethodGet, Path: p.path + "/file-1/content", StreamResponseBody: true,
			})
			if bifrostErr != nil {
				t.Fatalf("Passthrough: %+v", bifrostErr.Error)
			}
			if _, err := io.ReadFull(resp.BodyStream, make([]byte, 1024)); err != nil {
				t.Fatal(err)
			}
			closed := make(chan struct{})
			go func() {
				_ = resp.BodyStream.Close()
				close(closed)
			}()
			for _, ch := range []chan struct{}{closed, handlerDone} {
				select {
				case <-ch:
				case <-time.After(10 * time.Second):
					t.Fatal("Close drained the body instead of abandoning the connection")
				}
			}
		})
	}
}

// Providers whose passthrough must hold the whole body refuse the streamed fields before any
// upstream call: Vertex rewrites a JSON body's project and location (BIT-2941 keeps Vertex batch
// out), Bedrock and SageMaker sign the body. Gemini and Runware stream a request body but read
// the response whole. A custom provider inherits its base provider's support.
func TestStreamedPassthroughRefusedWhereUnsupported(t *testing.T) {
	const customBedrock = schemas.ModelProvider("custom-bedrock")
	cases := []struct {
		name     string
		provider schemas.ModelProvider
		req      *schemas.BifrostPassthroughRequest
	}{
		{"vertex body reader", schemas.Vertex, &schemas.BifrostPassthroughRequest{BodyReader: strings.NewReader("x"), BodySize: -1}},
		{"vertex response stream", schemas.Vertex, &schemas.BifrostPassthroughRequest{StreamResponseBody: true}},
		{"bedrock body reader", schemas.Bedrock, &schemas.BifrostPassthroughRequest{BodyReader: strings.NewReader("x"), BodySize: -1}},
		{"sagemaker response stream", schemas.SageMaker, &schemas.BifrostPassthroughRequest{StreamResponseBody: true}},
		{"gemini response stream", schemas.Gemini, &schemas.BifrostPassthroughRequest{StreamResponseBody: true}},
		{"runware response stream", schemas.Runware, &schemas.BifrostPassthroughRequest{StreamResponseBody: true}},
		{"custom on bedrock body reader", customBedrock, &schemas.BifrostPassthroughRequest{BodyReader: strings.NewReader("x"), BodySize: -1}},
	}
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { hits.Add(1) }))
	t.Cleanup(srv.Close)
	account := NewMockAccount()
	for _, provider := range []schemas.ModelProvider{schemas.Vertex, schemas.Bedrock, schemas.SageMaker, schemas.Gemini, schemas.Runware, customBedrock} {
		configureUpstream(account, provider, "key-"+string(provider), srv.URL, 0)
	}
	account.SetCustomProviderConfig(customBedrock, &schemas.CustomProviderConfig{BaseProviderType: schemas.Bedrock})
	client, err := Init(context.Background(), schemas.BifrostConfig{Account: account, Logger: NewDefaultLogger(schemas.LogLevelError)})
	if err != nil {
		t.Fatalf("Init: %v", err)
	}
	t.Cleanup(client.Shutdown)
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tc.req.Method, tc.req.Path = http.MethodPost, "/v1/files"
			_, bifrostErr := client.Passthrough(newPassthroughContext(t), tc.provider, tc.req)
			if bifrostErr == nil || bifrostErr.StatusCode == nil || *bifrostErr.StatusCode != http.StatusBadRequest {
				t.Fatalf("want a 400, got %+v", bifrostErr)
			}
			if hits.Load() != 0 {
				t.Fatal("the upstream was called")
			}
		})
	}
}

// A BodyReader request carries no request bytes to read usage or a raw request from, so the
// provider skips usage extraction: the caller bills from its own copy of the body. The control
// sends the same exchange with a byte Body, so the skip is not vacuous.
func TestStreamedPassthroughSkipsUsageExtraction(t *testing.T) {
	cases := []struct {
		provider schemas.ModelProvider
		path     string
		answer   string
	}{
		{schemas.OpenAI, "/v1/chat/completions", `{"object":"chat.completion","usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}`},
		{schemas.Anthropic, "/v1/messages", `{"type":"message","model":"claude-x","usage":{"input_tokens":3,"output_tokens":4}}`},
	}
	for _, tc := range cases {
		t.Run(string(tc.provider), func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, tc.answer)
			}))
			t.Cleanup(srv.Close)
			client := newStreamedPassthroughClient(t, tc.provider, srv.URL, 0)
			body := `{"model":"m","messages":[]}`

			control, bifrostErr := client.Passthrough(newPassthroughContext(t), tc.provider, &schemas.BifrostPassthroughRequest{
				Method: http.MethodPost, Path: tc.path, Body: []byte(body),
			})
			if bifrostErr != nil || control.PassthroughUsage == nil {
				t.Fatalf("control: usage %v, err %+v", control, bifrostErr)
			}
			resp, bifrostErr := client.Passthrough(newPassthroughContext(t), tc.provider, &schemas.BifrostPassthroughRequest{
				Method: http.MethodPost, Path: tc.path, BodyReader: strings.NewReader(body), BodySize: -1,
			})
			if bifrostErr != nil {
				t.Fatalf("Passthrough: %+v", bifrostErr.Error)
			}
			if resp.PassthroughUsage != nil {
				t.Fatalf("usage extracted from a reader body: %+v", resp.PassthroughUsage)
			}
		})
	}
}

// Usage skipped for a reader body is only what reads the request bytes: a Gemini Interactions
// create reports its run (job and served model) from the response alone, and the run is settled
// from that job, so it must survive a BodyReader.
func TestStreamedPassthroughKeepsInteractionsJob(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"int_run","agent":"deep-research-preview-04-2026","status":"in_progress"}`)
	}))
	t.Cleanup(srv.Close)
	client := newStreamedPassthroughClient(t, schemas.Gemini, srv.URL, 0)

	resp, bifrostErr := client.Passthrough(newPassthroughContext(t), schemas.Gemini, &schemas.BifrostPassthroughRequest{
		Method: http.MethodPost, Path: "/interactions", APIVersion: "v1beta",
		BodyReader: strings.NewReader(`{"agent":"deep-research-preview-04-2026","background":true}`), BodySize: -1,
	})
	if bifrostErr != nil {
		t.Fatalf("Passthrough: %+v", bifrostErr.Error)
	}
	if resp.ProviderJob == nil || resp.ProviderJob.ID != "int_run" || resp.ServedModel != "deep-research-preview-04-2026" {
		t.Fatalf("job %+v, served model %q", resp.ProviderJob, resp.ServedModel)
	}
}

// Body and BodyReader are two sources for one body; a request with both would be sent from one
// and billed or logged from the other, so it is refused before any upstream call.
func TestStreamedPassthroughRefusesBodyAndBodyReader(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { hits.Add(1) }))
	t.Cleanup(srv.Close)
	client := newStreamedPassthroughClient(t, schemas.OpenAI, srv.URL, 0)
	_, bifrostErr := client.Passthrough(newPassthroughContext(t), schemas.OpenAI, &schemas.BifrostPassthroughRequest{
		Method: http.MethodPost, Path: "/v1/files", Body: []byte("a"), BodyReader: strings.NewReader("b"), BodySize: -1,
	})
	if bifrostErr == nil || bifrostErr.StatusCode == nil || *bifrostErr.StatusCode != http.StatusBadRequest {
		t.Fatalf("want a 400, got %+v", bifrostErr)
	}
	if hits.Load() != 0 {
		t.Fatal("the upstream was called")
	}
}

// Fallbacks would re-send a drained BodyReader. A passthrough request carries none and no hook
// can attach any, so shouldTryFallbacks always sees an empty list.
func TestPassthroughHasNoFallbacks(t *testing.T) {
	req := &schemas.BifrostRequest{PassthroughRequest: &schemas.BifrostPassthroughRequest{BodyReader: strings.NewReader("x")}}
	req.SetFallbacks([]schemas.Fallback{{Provider: schemas.Azure, Model: "m"}})
	if _, _, fallbacks := req.GetRequestFields(); len(fallbacks) != 0 {
		t.Fatalf("passthrough request has fallbacks %v", fallbacks)
	}
}

// PassthroughStream frames SSE from a byte body; the streamed fields belong to Passthrough.
func TestPassthroughStreamRefusesStreamedFields(t *testing.T) {
	client := newStreamedPassthroughClient(t, schemas.OpenAI, "http://127.0.0.1:1", 0)
	for _, req := range []*schemas.BifrostPassthroughRequest{
		{Method: http.MethodPost, Path: "/v1/responses", BodyReader: strings.NewReader("x"), BodySize: -1},
		{Method: http.MethodPost, Path: "/v1/responses", StreamResponseBody: true},
	} {
		_, bifrostErr := client.PassthroughStream(newPassthroughContext(t), schemas.OpenAI, req)
		if bifrostErr == nil || bifrostErr.StatusCode == nil || *bifrostErr.StatusCode != http.StatusBadRequest {
			t.Fatalf("want a 400, got %+v", bifrostErr)
		}
	}
}

// Benchmarks the streamed upload and download through a real Bifrost: B/op and allocs/op
// must not grow with the body (compare the sizes), since the body is never held whole.
func BenchmarkStreamedPassthrough(b *testing.B) {
	for _, size := range []int{1 << 20, 32 << 20} {
		chunk := bytes.Repeat([]byte("y"), 64*1024)
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodPost {
				_, _ = io.Copy(io.Discard, r.Body)
				_, _ = io.WriteString(w, `{"id":"file-1"}`)
				return
			}
			// A fixed length keeps net/http's per-chunk framing allocations, which are the
			// server's, out of the client's B/op.
			w.Header().Set("Content-Length", strconv.Itoa(size))
			bw := bufio.NewWriter(w)
			for sent := 0; sent < size; sent += len(chunk) {
				_, _ = bw.Write(chunk)
			}
			_ = bw.Flush()
		}))
		account := NewMockAccount()
		configureUpstream(account, schemas.OpenAI, "key-openai", srv.URL, 0)
		client, err := Init(context.Background(), schemas.BifrostConfig{Account: account, Logger: NewDefaultLogger(schemas.LogLevelError)})
		if err != nil {
			b.Fatal(err)
		}

		b.Run("upload/"+sizeLabel(size), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(size))
			for i := 0; i < b.N; i++ {
				ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
				_, bifrostErr := client.Passthrough(ctx, schemas.OpenAI, &schemas.BifrostPassthroughRequest{
					Method: http.MethodPost, Path: "/v1/files",
					BodyReader: io.LimitReader(repeatReader{}, int64(size)), BodySize: -1,
				})
				if bifrostErr != nil {
					b.Fatal(bifrostErr.Error)
				}
			}
		})
		b.Run("download/"+sizeLabel(size), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(size))
			buf := make([]byte, 32*1024)
			for i := 0; i < b.N; i++ {
				ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
				resp, bifrostErr := client.Passthrough(ctx, schemas.OpenAI, &schemas.BifrostPassthroughRequest{
					Method: http.MethodGet, Path: "/v1/files/file-1/content", StreamResponseBody: true,
				})
				if bifrostErr != nil {
					b.Fatal(bifrostErr.Error)
				}
				if _, err := io.CopyBuffer(io.Discard, struct{ io.Reader }{resp.BodyStream}, buf); err != nil {
					b.Fatal(err)
				}
				_ = resp.BodyStream.Close()
			}
		})
		client.Shutdown()
		srv.Close()
	}
}

// Benchmarks the buffered passthrough (byte Body in, Body out) that every existing caller takes,
// so the streamed fields' cost on it shows up as an allocs/op or ns/op delta against the base.
func BenchmarkBufferedPassthrough(b *testing.B) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"file-1","object":"file","bytes":120}`)
	}))
	defer srv.Close()
	account := NewMockAccount()
	configureUpstream(account, schemas.OpenAI, "key-openai", srv.URL, 0)
	client, err := Init(context.Background(), schemas.BifrostConfig{Account: account, Logger: NewDefaultLogger(schemas.LogLevelError)})
	if err != nil {
		b.Fatal(err)
	}
	defer client.Shutdown()
	body := bytes.Repeat([]byte(`{"custom_id":"r","body":{"model":"m"}}`+"\n"), 16)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
		resp, bifrostErr := client.Passthrough(ctx, schemas.OpenAI, &schemas.BifrostPassthroughRequest{
			Method: http.MethodPost, Path: "/v1/files", Body: body,
		})
		if bifrostErr != nil || resp.StatusCode != http.StatusOK {
			b.Fatal(bifrostErr)
		}
	}
}

type repeatReader struct{}

func (repeatReader) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = 'z'
	}
	return len(p), nil
}

func sizeLabel(n int) string { return fmt.Sprintf("%dMiB", n>>20) }

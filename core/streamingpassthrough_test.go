package bifrost

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/providers/bedrock"
	"github.com/maximhq/bifrost/core/providers/sagemaker"
	"github.com/maximhq/bifrost/core/providers/vertex"
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

// Providers whose passthrough must hold the whole body refuse a BodyReader instead of sending
// something else: Vertex rewrites a JSON body's project and location (BIT-2941 keeps Vertex
// batch out), Bedrock and SageMaker sign the body.
func TestStreamedPassthroughRefusedWhereUnsupported(t *testing.T) {
	cases := []struct {
		name     string
		provider schemas.ModelProvider
		req      *schemas.BifrostPassthroughRequest
	}{
		{"vertex body reader", schemas.Vertex, &schemas.BifrostPassthroughRequest{BodyReader: strings.NewReader("x"), BodySize: -1}},
		{"bedrock body reader", schemas.Bedrock, &schemas.BifrostPassthroughRequest{BodyReader: strings.NewReader("x"), BodySize: -1}},
		{"sagemaker body reader", schemas.SageMaker, &schemas.BifrostPassthroughRequest{BodyReader: strings.NewReader("x"), BodySize: -1}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var hits atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { hits.Add(1) }))
			t.Cleanup(srv.Close)
			provider := newPassthroughProvider(t, tc.provider, srv.URL)
			tc.req.Method, tc.req.Path = http.MethodPost, "/v1/files"
			_, bifrostErr := provider.Passthrough(newPassthroughContext(t), passthroughTestKey(), tc.req)
			if bifrostErr == nil || bifrostErr.StatusCode == nil || *bifrostErr.StatusCode != http.StatusBadRequest {
				t.Fatalf("want a 400, got %+v", bifrostErr)
			}
			if hits.Load() != 0 {
				t.Fatal("the upstream was called")
			}
		})
	}
}

// PassthroughStream frames SSE from a byte body; BodyReader belongs to Passthrough.
func TestPassthroughStreamRefusesStreamedFields(t *testing.T) {
	client := newStreamedPassthroughClient(t, schemas.OpenAI, "http://127.0.0.1:1", 0)
	for _, req := range []*schemas.BifrostPassthroughRequest{
		{Method: http.MethodPost, Path: "/v1/responses", BodyReader: strings.NewReader("x"), BodySize: -1},
	} {
		_, bifrostErr := client.PassthroughStream(newPassthroughContext(t), schemas.OpenAI, req)
		if bifrostErr == nil || bifrostErr.StatusCode == nil || *bifrostErr.StatusCode != http.StatusBadRequest {
			t.Fatalf("want a 400, got %+v", bifrostErr)
		}
	}
}

// Benchmarks the streamed upload through a real Bifrost: B/op and allocs/op must not grow with
// the body (compare the sizes), since the body is never held whole.
func BenchmarkStreamedPassthrough(b *testing.B) {
	for _, size := range []int{1 << 20, 32 << 20} {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = io.Copy(io.Discard, r.Body)
			_, _ = io.WriteString(w, `{"id":"file-1"}`)
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
		client.Shutdown()
		srv.Close()
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

type passthroughProvider interface {
	Passthrough(*schemas.BifrostContext, schemas.Key, *schemas.BifrostPassthroughRequest) (*schemas.BifrostPassthroughResponse, *schemas.BifrostError)
}

// newPassthroughProvider builds the provider directly: the refusal comes before any key or
// credential is used, so no account plumbing is needed.
func newPassthroughProvider(t *testing.T, provider schemas.ModelProvider, url string) passthroughProvider {
	t.Helper()
	config := &schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{BaseURL: url, DefaultRequestTimeoutInSeconds: 5}}
	config.CheckAndSetDefaults()
	logger := NewDefaultLogger(schemas.LogLevelError)
	var (
		p   passthroughProvider
		err error
	)
	switch provider {
	case schemas.Vertex:
		p, err = vertex.NewVertexProvider(config, logger)
	case schemas.Bedrock:
		p, err = bedrock.NewBedrockProvider(config, logger)
	case schemas.SageMaker:
		p, err = sagemaker.NewSageMakerProvider(config, logger)
	default:
		t.Fatalf("no constructor for %s", provider)
	}
	if err != nil {
		t.Fatalf("new %s provider: %v", provider, err)
	}
	return p
}

func passthroughTestKey() schemas.Key {
	return schemas.Key{ID: "k", Value: *schemas.NewSecretVar("sk-test"), Models: schemas.WhiteList{"*"}}
}

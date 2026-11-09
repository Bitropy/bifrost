package utils

import (
	"context"
	"crypto/tls"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

const tornSSE = "data: {\"type\":\"content_block_delta\"}\n\n"

// tornStreamResponse serves tornSSE as a chunked body (over TLS or plain HTTP)
// and returns the streamed response. torn closes the connection after the
// event without the terminating 0 chunk, as a proxy (Envoy, a provider LB)
// does when its upstream resets mid-stream: a clean close, close_notify + FIN
// on TLS. Otherwise net/http ends the body with its terminating chunk.
func tornStreamResponse(t *testing.T, useTLS, torn bool) *fasthttp.Response {
	t.Helper()
	handler := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, tornSSE)
		w.(http.Flusher).Flush()
		if !torn {
			return
		}
		conn, _, err := w.(http.Hijacker).Hijack()
		if err != nil {
			t.Errorf("hijack: %v", err)
			return
		}
		_ = conn.Close()
	})
	var srv *httptest.Server
	if useTLS {
		srv = httptest.NewTLSServer(handler)
	} else {
		srv = httptest.NewServer(handler)
	}
	t.Cleanup(srv.Close)

	base := ConfigureDialer(&fasthttp.Client{
		ReadTimeout:  2 * time.Second,
		WriteTimeout: 2 * time.Second,
		TLSConfig:    &tls.Config{InsecureSkipVerify: true}, //nolint:gosec // httptest's self-signed certificate
	}, true)
	req := fasthttp.AcquireRequest()
	t.Cleanup(func() { fasthttp.ReleaseRequest(req) })
	resp := fasthttp.AcquireResponse()
	req.SetRequestURI(srv.URL)
	req.Header.SetMethod(http.MethodGet)
	resp.StreamBody = true
	if err := DoStreamingRequest(context.Background(), BuildStreamingClient(base), req, resp); err != nil {
		t.Fatalf("DoStreamingRequest: %v", err)
	}
	if resp.Header.ContentLength() != -1 {
		t.Fatalf("content length %d, want a chunked body (-1)", resp.Header.ContentLength())
	}
	return resp
}

// TestStreamingBody_TornChunkedStreamIsAnError: a chunked body closed before
// its terminating chunk reads as ErrStreamBodyCut (an io.ErrUnexpectedEOF),
// never as a clean io.EOF, or every stream loop finalizes a truncated answer
// as a success (BIT-2784). A body that does end with its terminating chunk
// still ends with io.EOF.
func TestStreamingBody_TornChunkedStreamIsAnError(t *testing.T) {
	for _, tc := range []struct {
		name      string
		tls, torn bool
	}{
		{"tls torn", true, true},
		{"plain torn", false, true},
		{"tls terminated", true, false},
		{"plain terminated", false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resp := tornStreamResponse(t, tc.tls, tc.torn)
			t.Cleanup(func() { fasthttp.ReleaseResponse(resp) })

			body, err := io.ReadAll(resp.BodyStream())
			if string(body) != tornSSE {
				t.Errorf("body %q, want %q", body, tornSSE)
			}
			if !tc.torn {
				if err != nil {
					t.Fatalf("terminated stream: err %v, want a clean end", err)
				}
				return
			}
			if !errors.Is(err, ErrStreamBodyCut) || !errors.Is(err, io.ErrUnexpectedEOF) {
				t.Fatalf("torn stream: err %v, want ErrStreamBodyCut (an io.ErrUnexpectedEOF)", err)
			}
		})
	}
}

// TestStreamPassthrough_TornChunkedStreamEndsWithTruncationError: the
// pass-through loop forwards what arrived, then ends a torn stream with the
// retryable 502 truncation error instead
// of its success chunk (BIT-2784). A terminated stream
// still ends with the success chunk.
func TestStreamPassthrough_TornChunkedStreamEndsWithTruncationError(t *testing.T) {
	for _, torn := range []bool{true, false} {
		t.Run(map[bool]string{true: "torn", false: "terminated"}[torn], func(t *testing.T) {
			resp := tornStreamResponse(t, true, torn)
			ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
			ch := StreamPassthrough(ctx, passthroughPostHook, nil, resp, resp.BodyStream(), PassthroughStreamParams{
				StatusCode: http.StatusOK,
				StartTime:  time.Now(),
				Observe: func([]byte) *schemas.BifrostPassthroughUsage {
					return &schemas.BifrostPassthroughUsage{LLMUsage: &schemas.BifrostLLMUsage{PromptTokens: 11}}
				},
			})
			var forwarded strings.Builder
			var last *schemas.BifrostStreamChunk
			for chunk := range ch {
				if chunk.BifrostPassthroughResponse != nil {
					forwarded.Write(chunk.BifrostPassthroughResponse.Body)
				}
				last = chunk
			}
			if forwarded.String() != tornSSE {
				t.Errorf("forwarded %q, want %q", forwarded.String(), tornSSE)
			}
			if last == nil {
				t.Fatal("no chunk")
			}
			if !torn {
				if last.BifrostError != nil || last.BifrostPassthroughResponse == nil || last.BifrostPassthroughResponse.PassthroughUsage == nil {
					t.Fatalf("terminated stream: last chunk %+v, want the success chunk with usage", last)
				}
				return
			}
			bifErr := last.BifrostError
			if bifErr == nil || bifErr.Error == nil || bifErr.Error.Message != schemas.ErrProviderStreamTruncated {
				t.Fatalf("torn stream: last chunk %+v, want the truncation error", last)
			}
			if bifErr.StatusCode == nil || *bifErr.StatusCode != http.StatusBadGateway || bifErr.IsBifrostError {
				t.Errorf("truncation error status %v IsBifrostError %v, want a retryable 502", bifErr.StatusCode, bifErr.IsBifrostError)
			}
		})
	}
}

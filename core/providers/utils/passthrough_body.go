package utils

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

// PassthroughErrorBodyLimit bounds the body read for a non-2xx answer to a StreamResponseBody
// request: an error envelope fits, and a provider that streams a large body with an error status
// cannot make the gateway buffer it.
const PassthroughErrorBodyLimit = 64 * 1024

var errPassthroughBodyAbandoned = errors.New("passthrough response body closed before EOF")

// SetPassthroughRequestBody puts the passthrough body on fasthttpReq: BodyReader as a stream when
// set, Body otherwise. Call it after the headers, since SetBodyStream owns Content-Length.
func SetPassthroughRequestBody(fasthttpReq *fasthttp.Request, req *schemas.BifrostPassthroughRequest) {
	if req.BodyReader == nil {
		fasthttpReq.SetBody(req.Body)
		return
	}
	size := int(req.BodySize)
	if req.BodySize < 0 {
		// fasthttp reads -2 as "identity until close", which a request cannot use.
		size = -1
	}
	// fasthttp writes no terminating chunk when the reader fails, so the upstream sees an
	// incomplete request instead of a shorter, valid one.
	fasthttpReq.SetBodyStream(req.BodyReader, size)
}

// RejectStreamedPassthrough refuses BodyReader and StreamResponseBody on a provider whose
// passthrough has to hold the whole body (a body rewrite or a request signature).
func RejectStreamedPassthrough(req *schemas.BifrostPassthroughRequest, provider schemas.ModelProvider) *schemas.BifrostError {
	if req.BodyReader == nil && !req.StreamResponseBody {
		return nil
	}
	return NewBifrostBadRequestError(fmt.Sprintf("streamed passthrough bodies are not supported by the %s provider", provider))
}

// RejectStreamedPassthroughResponse refuses StreamResponseBody on a provider whose passthrough
// streams the request body but must read the response whole.
func RejectStreamedPassthroughResponse(req *schemas.BifrostPassthroughRequest, provider schemas.ModelProvider) *schemas.BifrostError {
	if !req.StreamResponseBody {
		return nil
	}
	return NewBifrostBadRequestError(fmt.Sprintf("streamed passthrough response bodies are not supported by the %s provider", provider))
}

// DoPassthroughStreamResponse sends a passthrough request built by a provider and returns as soon
// as the response headers arrive. A 2xx body is handed back, gzip decoded, as BodyStream,
// guarded by the stream idle timeout and the request context; any other status is read into Body,
// bounded by PassthroughErrorBodyLimit. The caller keeps ownership of fasthttpReq.
func DoPassthroughStreamResponse(
	ctx *schemas.BifrostContext,
	streamingClient *fasthttp.Client,
	fasthttpReq *fasthttp.Request,
	path string,
	idleTimeoutSeconds int,
) (*schemas.BifrostPassthroughResponse, *schemas.BifrostError) {
	SetStreamIdleTimeoutIfEmpty(ctx, idleTimeoutSeconds)

	resp := fasthttp.AcquireResponse()
	resp.StreamBody = true
	activeClient := PrepareResponseStreaming(ctx, streamingClient, resp)

	startTime := time.Now()
	err := DoStreamingRequest(ctx, activeClient, fasthttpReq, resp)
	latency := time.Since(startTime)
	if err != nil {
		ReleaseStreamingResponse(ctx, resp)
		return nil, SetErrorLatency(streamingRequestError(err), latency)
	}

	status := resp.StatusCode()
	rawBody := resp.BodyStream()
	// Decoding before the header snapshot drops Content-Encoding from the returned headers.
	body, releaseDecoder := DecompressStreamBody(resp)
	headers := ExtractPassthroughProviderResponseHeaders(resp)
	ctx.SetValue(schemas.BifrostContextKeyProviderResponseHeaders, headers)

	out := &schemas.BifrostPassthroughResponse{
		StatusCode: status,
		Headers:    headers,
		ExtraFields: schemas.BifrostResponseExtraFields{
			Latency:                 latency.Milliseconds(),
			ProviderResponseHeaders: headers,
			PassthroughPath:         path,
		},
	}

	if status < 200 || status >= 300 || rawBody == nil {
		data, readErr := io.ReadAll(io.LimitReader(body, PassthroughErrorBodyLimit))
		releaseDecoder()
		ReleaseStreamingResponse(ctx, resp)
		if readErr != nil {
			return nil, NewBifrostOperationError("failed to read response body", readErr)
		}
		if status >= 200 && status < 300 {
			out.BodyStream = http.NoBody
		} else {
			out.Body = data
		}
		return out, nil
	}

	idle, stopIdle := NewIdleTimeoutReader(body, rawBody, GetStreamIdleTimeout(ctx), ctx)
	stopCancellation := SetupStreamCancellation(ctx, rawBody, nil)
	out.BodyStream = &passthroughBodyStream{
		reader: idle,
		release: func(eof bool) {
			stopCancellation()
			stopIdle()
			releaseDecoder()
			if eof {
				ReleaseStreamingResponse(ctx, resp)
				return
			}
			abandonStreamingResponse(ctx, rawBody)
		},
	}
	return out, nil
}

// abandonStreamingResponse closes a body that was not read to EOF instead of draining it:
// ReleaseStreamingResponse would read the rest of a body nobody wants, which for a batch file is
// up to hundreds of megabytes. Closing with an error keeps the connection out of the idle pool;
// the response is left to GC, as ReleaseStreamingResponse does on its own abandon paths.
func abandonStreamingResponse(ctx *schemas.BifrostContext, rawBody io.Reader) {
	if prev, _ := ctx.GetAndSetValue(schemas.BifrostContextKeyConnectionClosed, true).(bool); prev {
		return
	}
	defer func() {
		if rec := recover(); rec != nil {
			getLogger().Debug("recovered panic abandoning passthrough response body: %v", rec)
		}
	}()
	closeBodyStream(rawBody, errPassthroughBodyAbandoned)
}

// passthroughBodyStream is BifrostPassthroughResponse.BodyStream: Close frees the upstream
// response once, draining nothing when the body was not read to EOF.
type passthroughBodyStream struct {
	reader  io.Reader
	release func(eof bool)
	eof     atomic.Bool // Close may run on another goroutine than Read
	once    sync.Once
}

func (s *passthroughBodyStream) Read(p []byte) (int, error) {
	n, err := s.reader.Read(p)
	if err == io.EOF {
		s.eof.Store(true)
	}
	return n, err
}

func (s *passthroughBodyStream) Close() error {
	s.once.Do(func() { s.release(s.eof.Load()) })
	return nil
}

// streamingRequestError classifies a failed DoStreamingRequest the way the providers'
// PassthroughStream does.
func streamingRequestError(err error) *schemas.BifrostError {
	if errors.Is(err, context.Canceled) {
		return &schemas.BifrostError{
			IsBifrostError: false,
			Error: &schemas.ErrorField{
				Type:    schemas.Ptr(schemas.RequestCancelled),
				Message: schemas.ErrRequestCancelled,
				Error:   err,
			},
		}
	}
	if errors.Is(err, fasthttp.ErrTimeout) || errors.Is(err, context.DeadlineExceeded) {
		return NewBifrostTimeoutError(schemas.ErrProviderRequestTimedOut, err)
	}
	return NewBifrostOperationError(schemas.ErrProviderDoRequest, err)
}

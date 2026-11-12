package utils

import (
	"io"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

// PassthroughErrorBodyLimit bounds the body read for a non-2xx answer to a StreamResponseBody
// request: an error envelope fits, and a provider that streams a large body with an error status
// cannot make the gateway buffer it.
const PassthroughErrorBodyLimit = 64 * 1024

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

// DoPassthroughStreamResponse sends a passthrough request built by a provider and returns as soon
// as the response headers arrive. A 2xx body is handed back as BodyStream through
// WrapStreamingResponseBody; any other status is read into Body up to PassthroughErrorBodyLimit,
// and a longer body is abandoned, not drained. The caller keeps ownership of fasthttpReq.
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
	latency, bifrostErr, wait := MakeRequestWithContext(ctx, streamingClient, fasthttpReq, resp)
	wait()
	if bifrostErr != nil {
		ReleaseStreamingResponse(ctx, resp)
		return nil, bifrostErr
	}

	// Wrapping decodes the body, which drops Content-Encoding before the header snapshot.
	body := WrapStreamingResponseBody(ctx, resp)
	headers := ExtractPassthroughProviderResponseHeaders(resp)
	ctx.SetValue(schemas.BifrostContextKeyProviderResponseHeaders, headers)
	out := &schemas.BifrostPassthroughResponse{
		StatusCode: resp.StatusCode(),
		Headers:    headers,
		ExtraFields: schemas.BifrostResponseExtraFields{
			Latency:                 latency.Milliseconds(),
			ProviderResponseHeaders: headers,
			PassthroughPath:         path,
		},
	}
	if out.StatusCode >= 200 && out.StatusCode < 300 {
		out.BodyStream = body
		return out, nil
	}

	// One byte past the limit tells a longer body from one that ends at it: only the longer one
	// is left unread, and Close abandons it.
	data, readErr := io.ReadAll(io.LimitReader(body, PassthroughErrorBodyLimit+1))
	_ = body.Close()
	if readErr != nil {
		return nil, largeResponseReadError(readErr)
	}
	out.Body = data[:min(len(data), PassthroughErrorBodyLimit)]
	return out, nil
}

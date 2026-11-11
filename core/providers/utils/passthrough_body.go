package utils

import (
	"fmt"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

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

// RejectStreamedPassthrough refuses BodyReader on a provider whose passthrough has to hold the
// whole body (a body rewrite or a request signature).
func RejectStreamedPassthrough(req *schemas.BifrostPassthroughRequest, provider schemas.ModelProvider) *schemas.BifrostError {
	if req.BodyReader == nil {
		return nil
	}
	return NewBifrostBadRequestError(fmt.Sprintf("streamed passthrough bodies are not supported by the %s provider", provider))
}

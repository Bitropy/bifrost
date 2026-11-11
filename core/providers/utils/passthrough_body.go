package utils

import (
	"fmt"

	"github.com/maximhq/bifrost/core/schemas"
)

// RejectStreamedPassthrough refuses BodyReader on a provider whose passthrough has to hold the
// whole body (a body rewrite or a request signature).
func RejectStreamedPassthrough(req *schemas.BifrostPassthroughRequest, provider schemas.ModelProvider) *schemas.BifrostError {
	if req.BodyReader == nil {
		return nil
	}
	return NewBifrostBadRequestError(fmt.Sprintf("streamed passthrough bodies are not supported by the %s provider", provider))
}

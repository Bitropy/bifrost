package typesafe

import (
	"strings"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	schemas "github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

// parseTypesafeError parses a Typesafe error HTTP response into a BifrostError.
// Typesafe returns a JSON body detailing the issue on 401, 422, 429 and 529;
// the upstream status code and validation detail are preserved. The message is
// read in the order Typesafe's SDK extract_message uses: top-level
// message / error.message, then detail as a string, as an object
// {"error_type","message"} (the live API's shape), or as FastAPI's validation
// list [{"loc":[...],"msg":"..."}].
func parseTypesafeError(resp *fasthttp.Response) *schemas.BifrostError {
	var errorResp TypesafeError
	bifrostErr := providerUtils.HandleProviderAPIError(resp, &errorResp)

	message := errorResp.Message
	if message == "" && errorResp.Error != nil {
		message = errorResp.Error.Message
	}
	detailMessage, errorType := typesafeErrorDetail(errorResp.Detail)
	if message == "" {
		message = detailMessage
	}

	if bifrostErr.Error == nil {
		bifrostErr.Error = &schemas.ErrorField{}
	}
	if message != "" {
		bifrostErr.Error.Message = message
	} else if bifrostErr.Error.Message == "" {
		bifrostErr.Error.Message = "Typesafe API request failed"
	}
	if errorType != "" && bifrostErr.Error.Type == nil {
		bifrostErr.Error.Type = &errorType
	}

	return bifrostErr
}

// typesafeErrorDetail extracts the message and error type from the "detail"
// field of a Typesafe error body: a string, an {"error_type","message"} object,
// or a FastAPI validation list whose "msg" values are joined with "; ".
func typesafeErrorDetail(detail interface{}) (message, errorType string) {
	switch d := detail.(type) {
	case string:
		return d, ""
	case map[string]interface{}:
		message, _ = d["message"].(string)
		errorType, _ = d["error_type"].(string)
		return message, errorType
	case []interface{}:
		msgs := make([]string, 0, len(d))
		for _, item := range d {
			entry, ok := item.(map[string]interface{})
			if !ok {
				continue
			}
			if msg, _ := entry["msg"].(string); msg != "" {
				msgs = append(msgs, msg)
			}
		}
		return strings.Join(msgs, "; "), ""
	}
	return "", ""
}

// TypesafeNativeErrorDetail is the payload of Typesafe's native error body.
type TypesafeNativeErrorDetail struct {
	ErrorType string `json:"error_type"`
	Message   string `json:"message"`
}

// TypesafeNativeError is the error body Typesafe's API returns and its SDKs
// parse: {"detail": {"error_type": ..., "message": ...}} (observed on the live
// API; the docs describe JSON error bodies without pinning the schema).
type TypesafeNativeError struct {
	Detail TypesafeNativeErrorDetail `json:"detail"`
}

// ToTypesafeNativeError converts a Bifrost error into Typesafe's native error
// body so the /typesafe drop-in surface stays parseable by Typesafe's SDKs.
// The HTTP status code rides on the response as usual; this shapes the body.
func ToTypesafeNativeError(bifrostErr *schemas.BifrostError) *TypesafeNativeError {
	native := &TypesafeNativeError{
		Detail: TypesafeNativeErrorDetail{ErrorType: "api_error"},
	}
	if bifrostErr == nil {
		native.Detail.Message = "unknown error"
		return native
	}
	if bifrostErr.Error != nil {
		native.Detail.Message = bifrostErr.Error.Message
		if bifrostErr.Error.Type != nil && *bifrostErr.Error.Type != "" {
			native.Detail.ErrorType = *bifrostErr.Error.Type
		}
	}
	if native.Detail.Message == "" {
		native.Detail.Message = "Typesafe API request failed"
	}
	return native
}

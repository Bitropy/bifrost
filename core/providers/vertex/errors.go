package vertex

import (
	"cmp"
	"errors"
	"strings"
	"unicode/utf8"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/core/providers/gemini"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/tidwall/gjson"
	"github.com/valyala/fasthttp"
)

// parseVertexError converts a Vertex error response into a BifrostError, with the retry
// hint from a google.rpc.RetryInfo error detail or the response headers.
func parseVertexError(resp *fasthttp.Response) *schemas.BifrostError {
	bifrostErr := parseVertexErrorBody(resp)
	providerUtils.ApplyRetryAfter(bifrostErr, &resp.Header)
	return bifrostErr
}

// parseVertexErrorBody builds the error from the response body, with the retry hint when the
// body carries a google.rpc.RetryInfo detail.
func parseVertexErrorBody(resp *fasthttp.Response) *schemas.BifrostError {
	var openAIErr schemas.BifrostError
	var vertexErr []VertexError

	decodedBody, err := providerUtils.CheckAndDecodeBody(resp)
	if err != nil {
		bifrostErr := providerUtils.NewBifrostOperationError(schemas.ErrProviderResponseDecode, err)
		return bifrostErr
	}

	// Check for empty response
	trimmed := strings.TrimSpace(string(decodedBody))
	if len(trimmed) == 0 {
		bifrostErr := &schemas.BifrostError{
			IsBifrostError: false,
			StatusCode:     schemas.Ptr(resp.StatusCode()),
			Error: &schemas.ErrorField{
				Message: schemas.ErrProviderResponseEmpty,
			},
		}
		return bifrostErr
	}

	// Check for HTML error response before attempting JSON parsing
	if providerUtils.IsHTMLResponse(resp, decodedBody) {
		bifrostErr := &schemas.BifrostError{
			IsBifrostError: false,
			StatusCode:     schemas.Ptr(resp.StatusCode()),
			Error: &schemas.ErrorField{
				Message: schemas.ErrProviderResponseHTML,
				Error:   errors.New(string(decodedBody)),
			},
			ExtraFields: schemas.BifrostErrorExtraFields{
				RawResponse: string(decodedBody),
			},
		}
		return bifrostErr
	}

	createError := func(message, status string) *schemas.BifrostError {
		bifrostErr := providerUtils.NewProviderAPIError(message, nil, resp.StatusCode(), nil, nil)
		if status != "" {
			if bifrostErr.Error == nil {
				bifrostErr.Error = &schemas.ErrorField{}
			}
			bifrostErr.Error.Type = &status
		}
		var rawResponse interface{}
		if err := sonic.Unmarshal(decodedBody, &rawResponse); err != nil {
			rawResponse = string(decodedBody)
		}
		bifrostErr.ExtraFields.RawResponse = rawResponse
		return bifrostErr
	}

	// The OpenAI error envelope (Vertex's openapi endpoint, the Claude and
	// Llama publishers), with Vertex's status as the type when it carries none.
	if err := sonic.Unmarshal(decodedBody, &openAIErr); err == nil && openAIErr.Error != nil {
		openAIStatus := ""
		if openAIErr.Error.Type != nil {
			openAIStatus = *openAIErr.Error.Type
		}
		var single VertexError
		if openAIStatus == "" {
			if err := sonic.Unmarshal(decodedBody, &single); err == nil {
				openAIStatus = single.Error.Status
			}
		}
		message := openAIErr.Error.Message
		if message == "" {
			message = cmp.Or(publisherErrorMessage(decodedBody), "Unknown error")
		}
		bifrostErr := createError(message, openAIStatus)
		gemini.ApplyRetryInfo(bifrostErr, single.Error.Details)
		return bifrostErr
	}
	// Google's error envelope, as a list or a single object.
	if err := sonic.Unmarshal(decodedBody, &vertexErr); err == nil && len(vertexErr) > 0 && vertexErr[0].Error.Message != "" {
		bifrostErr := createError(vertexErr[0].Error.Message, vertexErr[0].Error.Status)
		gemini.ApplyRetryInfo(bifrostErr, vertexErr[0].Error.Details)
		return bifrostErr
	}
	var single VertexError
	if err := sonic.Unmarshal(decodedBody, &single); err == nil && single.Error.Message != "" {
		bifrostErr := createError(single.Error.Message, single.Error.Status)
		gemini.ApplyRetryInfo(bifrostErr, single.Error.Details)
		return bifrostErr
	}
	// A Google envelope without a message (a 429 RESOURCE_EXHAUSTED carrying
	// only RetryInfo) keeps its status type and retry hint below.
	status, details := single.Error.Status, single.Error.Details
	if len(vertexErr) > 0 && status == "" && len(details) == 0 {
		status, details = vertexErr[0].Error.Status, vertexErr[0].Error.Details
	}
	// A publisher's own error body (BIT-2765: Mistral's flat
	// {"object":"error","message":"…"}, its validation list) carries no Google
	// envelope; read its message rather than answering with an empty one.
	if message := publisherErrorMessage(decodedBody); message != "" {
		bifrostErr := createError(message, status)
		gemini.ApplyRetryInfo(bifrostErr, details)
		return bifrostErr
	}
	if gjson.ValidBytes(decodedBody) {
		bifrostErr := createError("Unknown error", status)
		gemini.ApplyRetryInfo(bifrostErr, details)
		return bifrostErr
	}
	// Not JSON at all: the upstream status and its text (capped) still reach
	// the caller, as the shared OpenAI-wire stream parser does, rather than a
	// status-less decode failure (which the client sees as a 500).
	return createError("provider API error: "+capErrorText(trimmed), "")
}

// maxErrorTextBytes caps the raw upstream text an error message carries.
const maxErrorTextBytes = 1024

// capErrorText is s cut to maxErrorTextBytes on a rune boundary, marked when cut.
func capErrorText(s string) string {
	if len(s) <= maxErrorTextBytes {
		return s
	}
	cut := maxErrorTextBytes
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut] + "…"
}

// publisherErrorMessage reads the message of an error body none of the Google
// or OpenAI envelopes describe: a root "message" string (Mistral's API error,
// {"object":"error","message":"…","type":"…","code":"…"}), a root "error"
// string, or the first "msg" of a FastAPI validation list, at the root
// ({"detail":[{"msg":…}]}) or under "message" (Mistral's 422). "" when none
// is present.
func publisherErrorMessage(body []byte) string {
	for _, path := range [...]string{"message", "error", "message.detail.0.msg", "detail.0.msg", "detail"} {
		if v := gjson.GetBytes(body, path); v.Type == gjson.String {
			if message := strings.TrimSpace(v.Str); message != "" {
				return message
			}
		}
	}
	return ""
}

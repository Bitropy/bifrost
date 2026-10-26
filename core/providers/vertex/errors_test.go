package vertex

import (
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/valyala/fasthttp"
)

// TestParseVertexError_PopulatesStatusType verifies the Vertex status
// (e.g. RESOURCE_EXHAUSTED) is surfaced on error.type rather than being dropped,
// so passthrough/OpenAI-shaped consumers see the exception type.
func TestParseVertexError_PopulatesStatusType(t *testing.T) {
	var resp fasthttp.Response
	resp.SetStatusCode(fasthttp.StatusTooManyRequests)
	resp.SetBodyString(`{"error":{"code":429,"message":"Quota exceeded","status":"RESOURCE_EXHAUSTED"}}`)

	bifrostErr := parseVertexError(&resp)

	require.NotNil(t, bifrostErr)
	require.NotNil(t, bifrostErr.Error)
	require.NotNil(t, bifrostErr.Error.Type, "nested error.type must be populated from status")
	assert.Equal(t, "RESOURCE_EXHAUSTED", *bifrostErr.Error.Type)
	assert.Equal(t, "Quota exceeded", bifrostErr.Error.Message)
}

// TestParseVertexError_NoStatusNoType verifies that when the body carries no
// Vertex status we don't fabricate an error.type.
func TestParseVertexError_NoStatusNoType(t *testing.T) {
	var resp fasthttp.Response
	resp.SetStatusCode(fasthttp.StatusBadRequest)
	resp.SetBodyString(`{"error":{"code":400,"message":"bad request"}}`)

	bifrostErr := parseVertexError(&resp)

	require.NotNil(t, bifrostErr)
	require.NotNil(t, bifrostErr.Error)
	assert.Nil(t, bifrostErr.Error.Type, "no status present, so none should be fabricated")
}

// TestParseVertexError_RetryInfo verifies a google.rpc.RetryInfo detail in the body
// becomes the error's retry hint, and that a body without one leaves it unset.
func TestParseVertexError_RetryInfo(t *testing.T) {
	var resp fasthttp.Response
	resp.SetStatusCode(fasthttp.StatusTooManyRequests)
	resp.SetBodyString(`{"error":{"code":429,"message":"Resource exhausted. Please try again later.","status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"12s"}]}}`)

	bifrostErr := parseVertexError(&resp)

	require.NotNil(t, bifrostErr)
	assert.Equal(t, int64(12000), bifrostErr.ExtraFields.RetryAfter)

	var plain fasthttp.Response
	plain.SetStatusCode(fasthttp.StatusTooManyRequests)
	plain.SetBodyString(`{"error":{"code":429,"message":"Quota exceeded","status":"RESOURCE_EXHAUSTED"}}`)
	assert.Equal(t, int64(0), parseVertexError(&plain).ExtraFields.RetryAfter)
}

// The cached-content and batch job parsers carry the retry hint too.
func TestVertexOperationErrors_RetryInfo(t *testing.T) {
	body := `{"error":{"code":429,"message":"Quota exceeded","status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"12s"}]}}`

	var cached fasthttp.Response
	cached.SetStatusCode(fasthttp.StatusTooManyRequests)
	cached.SetBodyString(body)
	cachedErr := parseVertexCachedContentError(&cached)
	assert.Equal(t, "Quota exceeded", cachedErr.Error.Message)
	assert.Equal(t, int64(12000), cachedErr.ExtraFields.RetryAfter)

	var job fasthttp.Response
	job.SetStatusCode(fasthttp.StatusTooManyRequests)
	job.Header.Set("Retry-After", "5")
	job.SetBodyString(`{"error":{"code":429,"message":"Quota exceeded"}}`)
	jobErr := parseVertexJobAPIError(&job, "batch create")
	assert.Equal(t, "Quota exceeded", jobErr.Error.Message)
	assert.Equal(t, int64(5000), jobErr.ExtraFields.RetryAfter)
}

// BIT-2765: a publisher's own error body (the Mistral publisher answers with
// Mistral's API error, not Google's envelope) keeps its status and its message
// on the unary path. Before, Mistral's flat body matched the single Vertex
// error shape with an empty message (a 400 with no reason), a validation
// list lost its msg the same way, and a body no shape described became a
// status-less decode failure, which the client saw as a 500.
func TestParseVertexError_PublisherBodies(t *testing.T) {
	for _, tc := range []struct {
		name, body, want string
	}{
		{"mistral", `{"object":"error","message":"Tool call id was call_weather_1 but must be a-z, A-Z, 0-9, with a length of 9.","type":"invalid_request_message_error","param":null,"code":"3280"}`, "Tool call id was call_weather_1 but must be a-z, A-Z, 0-9, with a length of 9."},
		{"mistral_numeric_code", `{"object":"error","message":"Invalid model","type":"invalid_model","param":null,"code":1500}`, "Invalid model"},
		{"mistral_422", `{"object":"error","message":{"detail":[{"type":"extra_forbidden","loc":["body","messages",1,"reasoning_content"],"msg":"Extra inputs are not permitted","input":"x"}]},"type":"invalid_request_error","param":null,"code":null}`, "Extra inputs are not permitted"},
		{"validation_list", `{"detail":[{"type":"missing","loc":["body","model"],"msg":"Field required","input":{}}]}`, "Field required"},
		{"detail_string", `{"detail":"Not authenticated"}`, "Not authenticated"},
		{"error_string", `{"error":"Tool call id was call_weather_1"}`, "Tool call id was call_weather_1"},
		{"plain_text", `Tool call id was call_weather_1 but must be a-z, A-Z, 0-9, with a length of 9.`, "provider API error: Tool call id was call_weather_1 but must be a-z, A-Z, 0-9, with a length of 9."},
		{"unknown_json_list", `[{"error":"x"}]`, "Unknown error"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var resp fasthttp.Response
			resp.SetStatusCode(fasthttp.StatusBadRequest)
			resp.SetBodyString(tc.body)

			bifrostErr := parseVertexError(&resp)

			require.NotNil(t, bifrostErr)
			require.NotNil(t, bifrostErr.StatusCode, "the upstream status is kept")
			assert.Equal(t, fasthttp.StatusBadRequest, *bifrostErr.StatusCode)
			assert.False(t, bifrostErr.IsBifrostError, "an upstream error, not a decode failure")
			require.NotNil(t, bifrostErr.Error)
			assert.Equal(t, tc.want, bifrostErr.Error.Message)
		})
	}
}

// A Google error envelope without a message keeps its status type and retry
// hint (it falls through to "Unknown error"), and an over-long non-JSON body
// is capped in the message.
func TestParseVertexError_EmptyMessageKeepsRetryInfoAndCapsText(t *testing.T) {
	var resp fasthttp.Response
	resp.SetStatusCode(fasthttp.StatusTooManyRequests)
	resp.SetBodyString(`{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"7s"}]}}`)
	bifrostErr := parseVertexError(&resp)
	require.NotNil(t, bifrostErr.Error)
	assert.Equal(t, "Unknown error", bifrostErr.Error.Message)
	require.NotNil(t, bifrostErr.Error.Type)
	assert.Equal(t, "RESOURCE_EXHAUSTED", *bifrostErr.Error.Type)
	assert.Equal(t, int64(7000), bifrostErr.ExtraFields.RetryAfter)

	var list fasthttp.Response
	list.SetStatusCode(fasthttp.StatusTooManyRequests)
	list.SetBodyString(`[{"error":{"code":429,"status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"3s"}]}}]`)
	assert.Equal(t, int64(3000), parseVertexError(&list).ExtraFields.RetryAfter)

	var long fasthttp.Response
	long.SetStatusCode(fasthttp.StatusBadRequest)
	long.SetBodyString(strings.Repeat("é", 2000))
	msg := parseVertexError(&long).Error.Message
	assert.LessOrEqual(t, len(msg), len("provider API error: ")+maxErrorTextBytes+len("…"))
	assert.True(t, utf8.ValidString(msg), "cut on a rune boundary")
	assert.Equal(t, fasthttp.StatusBadRequest, *parseVertexError(&long).StatusCode)
}

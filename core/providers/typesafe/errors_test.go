package typesafe

import (
	"net/http"
	"net/http/httptest"
	"testing"

	schemas "github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

func errorResponse(status int, body string) *fasthttp.Response {
	resp := fasthttp.AcquireResponse()
	resp.SetStatusCode(status)
	resp.Header.SetContentType("application/json")
	resp.SetBodyString(body)
	return resp
}

// TestParseTypesafeError pins the message forms TypeSafe's SDK extract_message
// reads, in its order: top-level error/message, then detail as str,
// dict.message or a FastAPI validation list.
func TestParseTypesafeError(t *testing.T) {
	tests := []struct {
		name        string
		status      int
		body        string
		wantMessage string
		wantType    string // "" means Type must stay unset
	}{
		{
			name:        "detail object with message and error_type",
			status:      422,
			body:        `{"detail":{"error_type":"validation_error","message":"questions.q1.type: unknown type"}}`,
			wantMessage: "questions.q1.type: unknown type",
			wantType:    "validation_error",
		},
		{
			name:        "detail object with only a message",
			status:      401,
			body:        `{"detail":{"message":"invalid API key"}}`,
			wantMessage: "invalid API key",
		},
		{
			name:        "detail list (FastAPI validation)",
			status:      422,
			body:        `{"detail":[{"loc":["body","state"],"msg":"field required","type":"missing"},{"loc":["body","questions"],"msg":"value is not a valid dict"}]}`,
			wantMessage: "field required; value is not a valid dict",
		},
		{
			name:        "detail string unchanged",
			status:      429,
			body:        `{"detail":"rate limit exceeded"}`,
			wantMessage: "rate limit exceeded",
		},
		{
			name:        "top-level message unchanged",
			status:      529,
			body:        `{"message":"overloaded"}`,
			wantMessage: "overloaded",
		},
		{
			name:        "top-level error.message wins over detail",
			status:      422,
			body:        `{"error":{"message":"top-level"},"detail":{"error_type":"validation_error","message":"from detail"}}`,
			wantMessage: "top-level",
			wantType:    "validation_error",
		},
		{
			name:        "empty body falls back",
			status:      422,
			body:        `{}`,
			wantMessage: "Typesafe API request failed",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			resp := errorResponse(tc.status, tc.body)
			t.Cleanup(func() { fasthttp.ReleaseResponse(resp) })

			bifrostErr := parseTypesafeError(resp)
			if bifrostErr == nil || bifrostErr.Error == nil {
				t.Fatalf("parseTypesafeError = %+v, want an error field", bifrostErr)
			}
			if got := bifrostErr.Error.Message; got != tc.wantMessage {
				t.Errorf("message = %q, want %q", got, tc.wantMessage)
			}
			if bifrostErr.StatusCode == nil || *bifrostErr.StatusCode != tc.status {
				t.Errorf("status = %v, want %d", bifrostErr.StatusCode, tc.status)
			}
			switch {
			case tc.wantType == "" && bifrostErr.Error.Type != nil:
				t.Errorf("type = %q, want unset", *bifrostErr.Error.Type)
			case tc.wantType != "" && (bifrostErr.Error.Type == nil || *bifrostErr.Error.Type != tc.wantType):
				t.Errorf("type = %v, want %q", bifrostErr.Error.Type, tc.wantType)
			}
		})
	}
}

// TestDecisionSurfacesUpstreamErrorDetail drives the object-shaped 422 through
// the provider's Decision: the upstream validation message must reach the
// caller instead of the generic fallback.
func TestDecisionSurfacesUpstreamErrorDetail(t *testing.T) {
	const upstreamMessage = "questions.q1.type: unknown type 'noulx'"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnprocessableEntity)
		_, _ = w.Write([]byte(`{"detail":{"error_type":"validation_error","message":"questions.q1.type: unknown type 'noulx'"}}`))
	}))
	t.Cleanup(server.Close)

	provider := newTestProvider(t, onPremConfig(server.URL, nil))
	_, bifrostErr := provider.Decision(testContext(), schemas.Key{}, noulRequest())
	if bifrostErr == nil || bifrostErr.Error == nil {
		t.Fatalf("Decision error = %+v, want an upstream error", bifrostErr)
	}
	if got := bifrostErr.Error.Message; got != upstreamMessage {
		t.Errorf("message = %q, want %q", got, upstreamMessage)
	}
	if bifrostErr.StatusCode == nil || *bifrostErr.StatusCode != http.StatusUnprocessableEntity {
		t.Errorf("status = %v, want 422", bifrostErr.StatusCode)
	}
	if bifrostErr.Error.Type == nil || *bifrostErr.Error.Type != "validation_error" {
		t.Errorf("type = %v, want validation_error", bifrostErr.Error.Type)
	}
}

package openai

import (
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

// A failed OpenAI-wire stream bills what the provider already reported, at
// the served tier (BIT-2741 F8-2): the chat loop's error event and read error
// carry the accumulated usage, and the Responses loop keeps a usage handle
// from the response object its events carry, so a response.failed with usage
// bills it.

type openAIErrorCapture struct {
	mu   sync.Mutex
	errs []*schemas.BifrostError
}

func (c *openAIErrorCapture) hook(_ *schemas.BifrostContext, resp *schemas.BifrostResponse, err *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError) {
	if err != nil {
		c.mu.Lock()
		c.errs = append(c.errs, err)
		c.mu.Unlock()
	}
	return resp, err
}

func (c *openAIErrorCapture) only(t *testing.T) *schemas.BifrostError {
	t.Helper()
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.errs) != 1 {
		t.Fatalf("expected exactly one stream error, got %d: %+v", len(c.errs), c.errs)
	}
	return c.errs[0]
}

func assertBilledPriority(t *testing.T, err *schemas.BifrostError, prompt, completion int) {
	t.Helper()
	billed := err.ExtraFields.BilledUsage
	if billed == nil {
		t.Fatalf("expected BilledUsage on %+v", err.Error)
	}
	if billed.PromptTokens != prompt || billed.CompletionTokens != completion {
		t.Errorf("billed = %d/%d, want %d/%d", billed.PromptTokens, billed.CompletionTokens, prompt, completion)
	}
	if billed.ServiceTier == nil || *billed.ServiceTier != schemas.BifrostServiceTierPriority {
		t.Errorf("billed service tier = %v, want priority", billed.ServiceTier)
	}
}

const openAITieredUsageChunk = `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"gpt-4.1","service_tier":"priority",` +
	`"choices":[{"index":0,"delta":{"content":"Hel"}}],"usage":{"prompt_tokens":12,"completion_tokens":3,"total_tokens":15}}` + "\n\n"

func TestChatStreamFailedExitsBillServedUsage(t *testing.T) {
	for _, tc := range []struct {
		name   string
		server func(t *testing.T) *httptest.Server
	}{
		{name: "error_event", server: func(t *testing.T) *httptest.Server {
			return completeSSEServer(t, openAITieredUsageChunk+`data: {"error":{"message":"The server had an error.","type":"server_error"}}`+"\n\n")
		}},
		{name: "read_error", server: func(t *testing.T) *httptest.Server {
			return truncatingSSEServer(t, openAITieredUsageChunk)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := tc.server(t)
			defer server.Close()
			capture := &openAIErrorCapture{}
			stream, bifrostErr := newStreamTestProvider(server.URL).ChatCompletionStream(newStreamTestContext(), capture.hook, nil, testKey(), basicChatRequest())
			if bifrostErr != nil {
				t.Fatalf("stream setup failed: %+v", bifrostErr)
			}
			collectChunks(t, stream)
			assertBilledPriority(t, capture.only(t), 12, 3)
		})
	}
}

func TestResponsesStreamFailedEventBillsServedUsage(t *testing.T) {
	body := "event: response.created\n" +
		`data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response","created_at":1,"status":"in_progress","model":"gpt-5","service_tier":"priority","output":[]}}` + "\n\n" +
		"event: response.failed\n" +
		`data: {"type":"response.failed","sequence_number":1,"response":{"id":"resp_1","object":"response","created_at":1,"status":"failed","model":"gpt-5",` +
		`"error":{"code":"server_error","message":"The server had an error."},"output":[],"usage":{"input_tokens":12,"output_tokens":3,"total_tokens":15}}}` + "\n\n"
	server := completeSSEServer(t, body)
	defer server.Close()
	capture := &openAIErrorCapture{}
	stream, bifrostErr := newStreamTestProvider(server.URL).ResponsesStream(newStreamTestContext(), capture.hook, nil, testKey(), &schemas.BifrostResponsesRequest{
		Provider: schemas.OpenAI,
		Model:    "gpt-5",
		Input: []schemas.ResponsesMessage{{
			Type:    schemas.Ptr(schemas.ResponsesMessageTypeMessage),
			Role:    schemas.Ptr(schemas.ResponsesInputMessageRoleUser),
			Content: &schemas.ResponsesMessageContent{ContentStr: schemas.Ptr("hi")},
		}},
	})
	if bifrostErr != nil {
		t.Fatalf("stream setup failed: %+v", bifrostErr)
	}
	collectChunks(t, stream)
	assertBilledPriority(t, capture.only(t), 12, 3)
}

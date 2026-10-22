package bedrock

import (
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/require"
)

// The converse stream bills its served tier and every error exit bills what
// the provider already reported (BIT-2741 F8-2): the metadata event's
// serviceTier rides the final chunk and the running usage handle, and an
// exception event after usage arrived carries that usage as BilledUsage.

type bedrockErrorCapture struct {
	mu   sync.Mutex
	errs []*schemas.BifrostError
}

func (c *bedrockErrorCapture) hook(_ *schemas.BifrostContext, resp *schemas.BifrostResponse, err *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError) {
	if err != nil {
		c.mu.Lock()
		c.errs = append(c.errs, err)
		c.mu.Unlock()
	}
	return resp, err
}

const converseTieredMetadata = `{"usage":{"inputTokens":12,"outputTokens":3,"totalTokens":15},"metrics":{"latencyMs":1},"serviceTier":{"type":"priority"}}`

func converseStreamServer(t *testing.T, write func(w http.ResponseWriter)) *httptest.Server {
	t.Helper()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		w.WriteHeader(http.StatusOK)
		write(w)
	}))
	t.Cleanup(ts.Close)
	return ts
}

func openConverseStream(t *testing.T, ts *httptest.Server, hook schemas.PostHookRunner) chan *schemas.BifrostStreamChunk {
	t.Helper()
	provider := newTestProviderWithServer(t, ts)
	req := testChatRequest()
	req.Model = testConverseStreamModel
	stream, bifrostErr := provider.ChatCompletionStream(testBedrockCtx(), hook, nil, testBedrockKey(), req)
	require.Nil(t, bifrostErr)
	return stream
}

func TestConverseStream_FinalChunkCarriesServedTier(t *testing.T) {
	ts := converseStreamServer(t, func(w http.ResponseWriter) {
		writeEventStreamEvent(t, w, "messageStart", []byte(`{"role":"assistant"}`))
		writeEventStreamEvent(t, w, "contentBlockDelta", []byte(`{"delta":{"text":"Hello"},"contentBlockIndex":0}`))
		writeEventStreamEvent(t, w, "messageStop", []byte(`{"stopReason":"end_turn"}`))
		writeEventStreamEvent(t, w, "metadata", []byte(converseTieredMetadata))
	})
	var last *schemas.BifrostChatResponse
	for chunk := range openConverseStream(t, ts, noopPostHookRunner) {
		require.Nil(t, chunk.BifrostError)
		if chunk.BifrostChatResponse != nil {
			last = chunk.BifrostChatResponse
		}
	}
	require.NotNil(t, last)
	require.NotNil(t, last.Usage)
	require.NotNil(t, last.ServiceTier, "the final chunk carries the served tier")
	require.Equal(t, schemas.BifrostServiceTierPriority, *last.ServiceTier)
}

func TestConverseStream_ExceptionAfterUsageBillsServedUsage(t *testing.T) {
	ts := converseStreamServer(t, func(w http.ResponseWriter) {
		writeEventStreamEvent(t, w, "messageStart", []byte(`{"role":"assistant"}`))
		writeEventStreamEvent(t, w, "metadata", []byte(converseTieredMetadata))
		writeEventStreamException(t, w, "throttlingException", "Too many requests")
	})
	capture := &bedrockErrorCapture{}
	for range openConverseStream(t, ts, capture.hook) {
	}
	require.Len(t, capture.errs, 1)
	billed := capture.errs[0].ExtraFields.BilledUsage
	require.NotNil(t, billed, "the exception carries the usage reported before it")
	require.Equal(t, 12, billed.PromptTokens)
	require.Equal(t, 3, billed.CompletionTokens)
	require.NotNil(t, billed.ServiceTier)
	require.Equal(t, schemas.BifrostServiceTierPriority, *billed.ServiceTier)
}

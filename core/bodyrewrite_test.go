package bifrost

import (
	"context"
	"fmt"
	"io"
	"testing"
	"time"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// rewriteAbortError is the BifrostError a provider builds from a transport
// failure that wraps providerUtils.ErrRequestBodyRewrite: an ordinary,
// otherwise-retried network error.
func rewriteAbortError() *schemas.BifrostError {
	return providerUtils.NewBifrostUpstreamConnectionError(schemas.ErrProviderDoRequest,
		fmt.Errorf("%w: token mutated", providerUtils.ErrRequestBodyRewrite))
}

// T1.6 (core half): an aborted rewrite is not retried, but fallbacks still run.
func TestExecuteRequestWithRetries_RequestBodyRewriteAbortNotRetried(t *testing.T) {
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	ctx.SetValue(schemas.BifrostContextKeyTracer, &schemas.NoOpTracer{})
	logger := NewDefaultLogger(schemas.LogLevelError)

	// Control: the same error without the sentinel is retried.
	plain := providerUtils.NewBifrostUpstreamConnectionError(schemas.ErrProviderDoRequest, io.ErrUnexpectedEOF)
	for _, tc := range []struct {
		name      string
		err       *schemas.BifrostError
		wantCalls int
	}{
		{"rewrite abort", rewriteAbortError(), 1},
		{"plain network error", plain, 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			handler := func(schemas.Key) (string, *schemas.BifrostError) {
				calls++
				return "", tc.err
			}
			_, err := executeRequestWithRetries(ctx, createTestConfig(2, time.Millisecond, time.Millisecond),
				handler, nil, schemas.ChatCompletionRequest, schemas.OpenAI, "gpt-4", nil, logger)
			require.Same(t, tc.err, err)
			assert.Equal(t, tc.wantCalls, calls)
		})
	}

	b := &Bifrost{logger: logger}
	req := &schemas.BifrostRequest{
		RequestType: schemas.ChatCompletionRequest,
		ChatRequest: &schemas.BifrostChatRequest{
			Provider:  schemas.OpenAI,
			Model:     "gpt-4",
			Fallbacks: []schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude"}},
		},
	}
	assert.True(t, b.shouldTryFallbacks(schemas.NewBifrostContext(context.Background(), schemas.NoDeadline), req, rewriteAbortError()), "an abort must still allow fallbacks")
}

// T1.9: internal requests drop the rewriter; fallbacks keep it.
func TestRequestBodyRewriterContextIsolation(t *testing.T) {
	var rw providerUtils.RequestBodyRewriter = func([]byte) (io.Reader, int64, error) { return nil, 0, nil }
	newCtx := func() *schemas.BifrostContext {
		ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
		ctx.SetValue(schemas.BifrostContextKeyRequestBodyRewriter, rw)
		ctx.SetValue(schemas.BifrostContextKeyIntegrationType, "anthropic")
		return ctx
	}

	internal := newCtx()
	ClearContextForInternalRequest(internal)
	assert.Nil(t, providerUtils.RequestBodyRewriterFromContext(internal))

	prepared := newCtx()
	PrepareContextForInternalRequest(prepared)
	assert.Nil(t, providerUtils.RequestBodyRewriterFromContext(prepared))

	fallback := newCtx()
	clearCtxForFallback(fallback)
	assert.NotNil(t, providerUtils.RequestBodyRewriterFromContext(fallback))

	converted := newCtx()
	clearAnthropicPassthroughForNonNativeProvider(converted, schemas.OpenAI, "gpt-4")
	assert.NotNil(t, providerUtils.RequestBodyRewriterFromContext(converted))
}

// BIT-2567 (spec D3, corrected): an aborted rewrite is a splice failure on the
// embedder's side, not a fault of the key that served the attempt. The retry
// loop ends it in the terminal clause without retrying, and stampEndedAttempt
// (which labels only timeouts and cancellations) leaves its record unclassified,
// so nothing that reads the trail (a load balancer, key telemetry) books it
// against the key.
func TestExecuteRequestWithRetries_RequestBodyRewriteAbortIsNotAKeyFault(t *testing.T) {
	ctx := rotationTestContext()
	abort := rewriteAbortError()
	calls := 0
	handler := func(schemas.Key) (string, *schemas.BifrostError) {
		calls++
		return "", abort
	}
	_, err := executeRequestWithRetries(ctx, createTestConfig(2, time.Millisecond, time.Millisecond),
		handler, poolKeyProvider([]schemas.Key{rotationKeyA, rotationKeyB}),
		schemas.ChatCompletionRequest, schemas.OpenAI, "gpt-4", nil, NewDefaultLogger(schemas.LogLevelError))
	require.Same(t, abort, err)
	require.Equal(t, 1, calls, "an aborted rewrite is neither retried nor rotated to another key")

	record := attemptTrail(t, ctx, 1)[0]
	assert.Equal(t, rotationKeyA.ID, record.KeyID)
	assert.Empty(t, record.FailureClass, "a rewrite abort must not be classified against the key: %+v", record)
	assert.False(t, record.FailureClass.IsPerKey(), "a rewrite abort must not carry a per-key class: %+v", record)
	assert.Nil(t, record.FailReason, "a rewrite abort must not carry a FailReason, read as \"this key failed\": %+v", record)
	assert.False(t, record.TriggeredRotation, "a rewrite abort must not rotate the key: %+v", record)
}

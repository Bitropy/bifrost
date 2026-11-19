package bifrost

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/maximhq/bifrost/core/providers/anthropic"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const refusalAnthropicMessage = `{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`

const refusalAnthropicStream = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-5\",\"content\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n" +
	"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
	"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n" +
	"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
	"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n" +
	"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"

// refusalUpstream serves Anthropic Messages, unary or SSE, and counts requests; status overrides
// the answer with that HTTP status.
func refusalUpstream(t *testing.T, status int) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		body, _ := io.ReadAll(r.Body)
		if status != 0 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(status)
			_, _ = io.WriteString(w, `{"type":"error","error":{"type":"api_error","message":"down"}}`)
			return
		}
		if strings.Contains(string(body), `"stream":true`) {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, refusalAnthropicStream)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, refusalAnthropicMessage)
	}))
	t.Cleanup(srv.Close)
	return srv, &hits
}

func refusalClient(t *testing.T, account *MockAccount) *Bifrost {
	t.Helper()
	client, err := Init(context.Background(), schemas.BifrostConfig{Account: account, Logger: NewDefaultLogger(schemas.LogLevelError)})
	require.NoError(t, err)
	t.Cleanup(client.Shutdown)
	return client
}

func refusalModeContext(refuse bool) *schemas.BifrostContext {
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	ctx.SetValue(schemas.BifrostContextKeyRequestBodyRewriter, &schemas.RequestBodyRewriteHook{RefuseFeatures: refuse})
	return ctx
}

func refusalChat(provider schemas.ModelProvider, model string, params *schemas.ChatParameters) *schemas.BifrostChatRequest {
	return &schemas.BifrostChatRequest{
		Provider: provider,
		Model:    model,
		Input:    []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("hi")}}},
		Params:   params,
	}
}

// AC-8: a refusal is returned before any network call and survives to the engine boundary, where
// errors.As finds the typed refusal on ErrorField.Error. Azure Claude carries no inference_geo.
func TestRefusalMode_RefusalReachesCallerWithoutUpstreamCall(t *testing.T) {
	srv, hits := refusalUpstream(t, 0)
	account := NewMockAccount()
	configureUpstream(account, schemas.Azure, "key-azure", srv.URL, 0)
	client := refusalClient(t, account)

	params := &schemas.ChatParameters{MaxCompletionTokens: schemas.Ptr(64), InferenceGeo: schemas.Ptr("us")}
	_, bifrostErr := client.ChatCompletionRequest(refusalModeContext(true), refusalChat(schemas.Azure, "claude-sonnet-4-5", params))
	require.NotNil(t, bifrostErr)
	require.NotNil(t, bifrostErr.Error)
	var refusal *schemas.UnsupportedOnTargetError
	require.True(t, errors.As(bifrostErr.Error.Error, &refusal), "got %s", bifrostErr.GetErrorString())
	assert.Equal(t, "inference_geo", refusal.Feature)
	assert.Equal(t, schemas.Azure, refusal.Provider)
	assert.Equal(t, 400, *bifrostErr.StatusCode)
	assert.Zero(t, hits.Load(), "no upstream call")

	_, bifrostErr = client.ChatCompletionStreamRequest(refusalModeContext(true), refusalChat(schemas.Azure, "claude-sonnet-4-5", params))
	require.NotNil(t, bifrostErr)
	require.True(t, errors.As(bifrostErr.Error.Error, &refusal))
	assert.Zero(t, hits.Load(), "no upstream call on the stream path")

	// Refusal mode off: the same request is served with the field stripped, as today.
	resp, bifrostErr := client.ChatCompletionRequest(refusalModeContext(false), refusalChat(schemas.Azure, "claude-sonnet-4-5", params))
	require.Nil(t, bifrostErr)
	require.NotNil(t, resp)
	assert.Equal(t, int32(1), hits.Load())
}

// AC-8: the attempt's adaptation set is copied onto the unary response and onto every stream
// chunk, starting from the bits the embedder put on the parent context.
func TestRefusalMode_AdaptationsOnResponseAndEveryChunk(t *testing.T) {
	srv, _ := refusalUpstream(t, 0)
	account := NewMockAccount()
	configureUpstream(account, schemas.Anthropic, "key-anthropic", srv.URL, 0)
	client := refusalClient(t, account)

	// No output cap (max_tokens) and a seed core ignores (unmapped); the embedder adds reasoning.
	params := &schemas.ChatParameters{Seed: schemas.Ptr(7)}
	want := schemas.AdaptationMaxTokens | schemas.AdaptationUnmapped | schemas.AdaptationReasoning
	newCtx := func() *schemas.BifrostContext {
		parent := context.WithValue(context.Background(), schemas.BifrostContextKeyAdaptations, schemas.AdaptationReasoning)
		ctx := schemas.NewBifrostContext(parent, schemas.NoDeadline)
		ctx.SetValue(schemas.BifrostContextKeyRequestBodyRewriter, &schemas.RequestBodyRewriteHook{RefuseFeatures: true})
		return ctx
	}

	resp, bifrostErr := client.ChatCompletionRequest(newCtx(), refusalChat(schemas.Anthropic, "claude-sonnet-4-5", params))
	require.Nil(t, bifrostErr)
	assert.Equal(t, want, resp.ExtraFields.Adaptations)

	stream, bifrostErr := client.ChatCompletionStreamRequest(newCtx(), refusalChat(schemas.Anthropic, "claude-sonnet-4-5", params))
	require.Nil(t, bifrostErr)
	chunks := 0
	for chunk := range stream {
		require.Nil(t, chunk.BifrostError)
		require.NotNil(t, chunk.BifrostChatResponse)
		assert.Equal(t, want, chunk.BifrostChatResponse.ExtraFields.Adaptations, "chunk %d", chunks)
		chunks++
	}
	assert.Greater(t, chunks, 1)

	// Nothing adapted: nothing is written.
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	resp, bifrostErr = client.ChatCompletionRequest(ctx, refusalChat(schemas.Anthropic, "claude-sonnet-4-5", &schemas.ChatParameters{MaxCompletionTokens: schemas.Ptr(64)}))
	require.Nil(t, bifrostErr)
	assert.Zero(t, resp.ExtraFields.Adaptations)
	assert.Nil(t, ctx.Value(schemas.BifrostContextKeyAdaptations))
}

// AC-8, attempt scope: a core call whose own fallback chain runs two provider attempts starts the
// second with both context sets empty. The first (Azure, adaptive-only model) removes
// inference_geo and adapts sampling, then fails upstream; the second (Anthropic) does neither.
func TestRefusalMode_FallbackAttemptStartsWithEmptySets(t *testing.T) {
	failing, failingHits := refusalUpstream(t, http.StatusInternalServerError)
	serving, servingHits := refusalUpstream(t, 0)
	account := NewMockAccount()
	configureUpstream(account, schemas.Azure, "key-azure", failing.URL, 0)
	configureUpstream(account, schemas.Anthropic, "key-anthropic", serving.URL, 0)
	client := refusalClient(t, account)

	req := refusalChat(schemas.Azure, "claude-opus-4-7", &schemas.ChatParameters{
		MaxCompletionTokens: schemas.Ptr(64), Temperature: schemas.Ptr(0.5), InferenceGeo: schemas.Ptr("us"),
	})
	req.Fallbacks = []schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-sonnet-4-5"}}
	ctx := refusalModeContext(false)
	resp, bifrostErr := client.ChatCompletionRequest(ctx, req)
	require.Nil(t, bifrostErr)
	require.Equal(t, int32(1), failingHits.Load(), "the first attempt reached its upstream")
	require.Equal(t, int32(1), servingHits.Load())
	assert.Zero(t, resp.ExtraFields.Adaptations, "the failed attempt's adaptations do not reach the served one")
	assert.Zero(t, anthropic.RemovedFeatures(ctx), "the failed attempt's removed features do not reach the served one")
	assert.Zero(t, schemas.AdaptationsFromContext(ctx))
}

// A core fallback drops the failed attempt's adaptations but keeps the ones the embedder recorded
// on the call's parent context: the embedder's edit holds for every attempt.
func TestRefusalMode_FallbackAttemptKeepsEmbedderAdaptations(t *testing.T) {
	failing, _ := refusalUpstream(t, http.StatusInternalServerError)
	serving, servingHits := refusalUpstream(t, 0)
	account := NewMockAccount()
	configureUpstream(account, schemas.Azure, "key-azure", failing.URL, 0)
	configureUpstream(account, schemas.Anthropic, "key-anthropic", serving.URL, 0)
	client := refusalClient(t, account)

	req := refusalChat(schemas.Azure, "claude-opus-4-7", &schemas.ChatParameters{
		MaxCompletionTokens: schemas.Ptr(64), Temperature: schemas.Ptr(0.5),
	})
	req.Fallbacks = []schemas.Fallback{{Provider: schemas.Anthropic, Model: "claude-sonnet-4-5"}}
	parent := context.WithValue(context.Background(), schemas.BifrostContextKeyAdaptations, schemas.AdaptationReasoning)
	ctx := schemas.NewBifrostContext(parent, schemas.NoDeadline)
	resp, bifrostErr := client.ChatCompletionRequest(ctx, req)
	require.Nil(t, bifrostErr)
	require.Equal(t, int32(1), servingHits.Load())
	assert.Equal(t, schemas.AdaptationReasoning, resp.ExtraFields.Adaptations)
}

// D-6: a hook with a nil Rewrite rewrites nothing; a bare rewriter still works.
func TestRequestBodyRewriteHook_NilRewriteSkipped(t *testing.T) {
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	ctx.SetValue(schemas.BifrostContextKeyRequestBodyRewriter, &schemas.RequestBodyRewriteHook{RefuseFeatures: true})
	assert.Nil(t, providerUtils.RequestBodyRewriterFromContext(ctx))
	assert.True(t, schemas.FeatureRefusal(ctx))

	called := false
	ctx.SetValue(schemas.BifrostContextKeyRequestBodyRewriter, &schemas.RequestBodyRewriteHook{Rewrite: func([]byte) (io.Reader, int64, error) {
		called = true
		return nil, 0, nil
	}})
	rw := providerUtils.RequestBodyRewriterFromContext(ctx)
	require.NotNil(t, rw)
	_, _, _ = rw(nil)
	assert.True(t, called)
	assert.False(t, schemas.FeatureRefusal(ctx))

	ctx.SetValue(schemas.BifrostContextKeyRequestBodyRewriter, providerUtils.RequestBodyRewriter(func([]byte) (io.Reader, int64, error) { return nil, 0, nil }))
	assert.NotNil(t, providerUtils.RequestBodyRewriterFromContext(ctx))
	assert.False(t, schemas.FeatureRefusal(ctx))

	// A nil-Rewrite hook is served end to end: the request is sent as built.
	srv, hits := refusalUpstream(t, 0)
	account := NewMockAccount()
	configureUpstream(account, schemas.Anthropic, "key-anthropic", srv.URL, 0)
	client := refusalClient(t, account)
	_, bifrostErr := client.ChatCompletionRequest(refusalModeContext(true), refusalChat(schemas.Anthropic, "claude-sonnet-4-5", &schemas.ChatParameters{MaxCompletionTokens: schemas.Ptr(64)}))
	require.Nil(t, bifrostErr)
	assert.Equal(t, int32(1), hits.Load())
}

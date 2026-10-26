package anthropic

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
)

// Claude Haiku 5.5 is adaptive-only (BIT-2738): thinking:{type:"enabled",
// budget_tokens:N} is a 400 ("thinking.type.enabled" is not supported for this
// model. Use "thinking.type.adaptive" and "output_config.effort"), seen on
// Vertex (claude-haiku-5-5) for chat, /v1/messages and /v1/responses. Its
// thinking surface is Opus 4.7+'s: adaptive on by default, effort low-max,
// non-default sampling rejected, and "disabled" only at effort high or below.
//
// Source: https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking
// (per-model thinking table, Claude Haiku 5.5 row).

// haiku5Models are the ids Haiku 5.5 is addressed by: first party, Vertex
// (bare and @-versioned), Bedrock (regional and global inference profiles).
var haiku5Models = []string{
	"claude-haiku-5-5",
	"claude-haiku-5-5@20261001",
	"anthropic.claude-haiku-5-5",
	"global.anthropic.claude-haiku-5-5-v1:0",
	"us.anthropic.claude-haiku-5-5",
}

func TestHaiku5_Predicates(t *testing.T) {
	for _, m := range haiku5Models {
		caps := schemas.ResolveModelCaps(schemas.Anthropic, m)
		assert.True(t, IsHaiku5Plus(m), m)
		assert.True(t, caps.AdaptiveOnlyThinking(DefaultAdaptiveOnlyThinking(m)), "%s is adaptive-only", m)
		assert.True(t, RejectsEnabledThinking(caps), "%s rejects enabled", m)
		assert.True(t, caps.SupportsAdaptiveThinking(DefaultSupportsAdaptiveThinking(m)), "%s accepts adaptive", m)
		assert.True(t, DefaultSupportsNativeEffort(m), "%s takes output_config.effort", m)
		assert.False(t, SupportsNativeEffort(caps), "%s takes effort with adaptive, not with a budget", m)
		assert.Equal(t, []string{"low", "medium", "high", "xhigh", "max"}, DefaultEffortControl(m).Levels, m)
		assert.False(t, RejectsDisabledThinking(caps, new("high")), "%s accepts disabled at high", m)
		assert.True(t, RejectsDisabledThinking(caps, new("xhigh")), "%s rejects disabled at xhigh", m)
		assert.True(t, RejectsDisabledThinking(caps, new("max")), "%s rejects disabled at max", m)
	}
	for _, m := range []string{"claude-haiku-4-5", "claude-haiku-4-5-20251001", "anthropic.claude-haiku-4-5-20251001-v1:0", "claude-3-5-haiku"} {
		caps := schemas.ResolveModelCaps(schemas.Anthropic, m)
		assert.False(t, IsHaiku5Plus(m), m)
		assert.False(t, caps.AdaptiveOnlyThinking(DefaultAdaptiveOnlyThinking(m)), "%s keeps budget_tokens", m)
		assert.False(t, caps.SupportsAdaptiveThinking(DefaultSupportsAdaptiveThinking(m)), m)
		assert.False(t, DefaultSupportsNativeEffort(m), m)
	}
}

// The typed chat converter (managed chat, and /v1/messages and /v1/responses
// once core converts them) sends adaptive thinking with the effort, no budget
// and no sampling parameter, for an effort ask and a budget ask alike.
func TestHaiku5_ChatConverterSendsAdaptive(t *testing.T) {
	for _, reasoning := range []*schemas.ChatReasoning{
		{Effort: new("high")},
		{MaxTokens: new(2048)},
	} {
		req := &schemas.BifrostChatRequest{
			Provider: schemas.Vertex,
			Model:    "claude-haiku-5-5",
			Input:    []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: new("think")}}},
			Params:   &schemas.ChatParameters{MaxCompletionTokens: new(8192), Temperature: new(0.3), Reasoning: reasoning},
		}
		ctx, cancel := schemas.NewBifrostContextWithCancel(context.Background())
		out, err := ToAnthropicChatRequest(ctx, req)
		cancel()
		require.NoError(t, err)
		require.NotNil(t, out.Thinking)
		assert.Equal(t, "adaptive", out.Thinking.Type)
		assert.Nil(t, out.Thinking.BudgetTokens)
		assert.Nil(t, out.Temperature, "Haiku 5.5 rejects a non-default temperature")
		if reasoning.Effort != nil {
			require.NotNil(t, out.OutputConfig)
			require.NotNil(t, out.OutputConfig.Effort)
			assert.Equal(t, "high", *out.OutputConfig.Effort)
		}
	}
}

// A client's own Anthropic thinking block (a /v1/messages body) is rewritten to
// adaptive on the typed request and on a raw body alike, and adaptive is never
// rewritten back to enabled (the pre-adaptive Haiku 4.5 rule).
func TestHaiku5_ClientThinkingBecomesAdaptive(t *testing.T) {
	for _, m := range haiku5Models {
		body := []byte(`{"model":"` + m + `","max_tokens":4096,"thinking":{"type":"enabled","budget_tokens":2048}}`)
		out, err := StripUnsupportedFieldsFromRawBody(body, schemas.Anthropic, m)
		require.NoError(t, err)
		assert.Equal(t, "adaptive", providerUtils.GetJSONField(out, "thinking.type").String(), "%s: %s", m, out)
		assert.False(t, providerUtils.JSONFieldExists(out, "thinking.budget_tokens"), "%s: %s", m, out)

		adaptive := []byte(`{"model":"` + m + `","max_tokens":4096,"thinking":{"type":"adaptive"},"output_config":{"effort":"xhigh"}}`)
		out, err = StripUnsupportedFieldsFromRawBody(adaptive, schemas.Anthropic, m)
		require.NoError(t, err)
		assert.Equal(t, "adaptive", providerUtils.GetJSONField(out, "thinking.type").String(), "%s: %s", m, out)
		assert.Equal(t, "xhigh", providerUtils.GetJSONField(out, "output_config.effort").String(), "%s: %s", m, out)
	}
}

// The Responses converter (/v1/responses on a Claude deployment) sends the
// same shape.
func TestHaiku5_ResponsesConverterSendsAdaptive(t *testing.T) {
	req := &schemas.BifrostResponsesRequest{
		Provider: schemas.Vertex,
		Model:    "claude-haiku-5-5",
		Input: []schemas.ResponsesMessage{{
			Role:    schemas.Ptr(schemas.ResponsesInputMessageRoleUser),
			Content: &schemas.ResponsesMessageContent{ContentStr: new("think")},
		}},
		Params: &schemas.ResponsesParameters{MaxOutputTokens: new(8192), Reasoning: &schemas.ResponsesParametersReasoning{Effort: new("medium")}},
	}
	ctx, cancel := schemas.NewBifrostContextWithCancel(context.Background())
	defer cancel()
	out, err := ToAnthropicResponsesRequest(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, out.Thinking)
	assert.Equal(t, "adaptive", out.Thinking.Type)
	assert.Nil(t, out.Thinking.BudgetTokens)
	require.NotNil(t, out.OutputConfig)
	require.NotNil(t, out.OutputConfig.Effort)
	assert.Equal(t, "medium", *out.OutputConfig.Effort)
}

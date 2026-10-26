package bedrock

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/maximhq/bifrost/core/providers/anthropic"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
)

// Claude Haiku 5.5 is adaptive-only on Bedrock too (BIT-2738): Converse and
// InvokeModel send thinking:{type:"adaptive"} with the effort, never
// {type:"enabled",budget_tokens:N}, which the model refuses with a 400.
const haiku5Model = "global.anthropic.claude-haiku-5-5"

func TestHaiku5_ConverseSendsAdaptive(t *testing.T) {
	req := &schemas.BifrostChatRequest{
		Model:  haiku5Model,
		Params: &schemas.ChatParameters{Reasoning: &schemas.ChatReasoning{Effort: schemas.Ptr("high")}},
	}
	out := &BedrockConverseRequest{}
	require.NoError(t, convertChatParameters(nil, req, out, schemas.ResolveModelCaps(schemas.Bedrock, haiku5Model)))
	require.NotNil(t, out.AdditionalModelRequestFields)
	raw, ok := out.AdditionalModelRequestFields.Get("thinking")
	require.True(t, ok)
	thinking, ok := raw.(map[string]any)
	require.True(t, ok, "%T", raw)
	assert.Equal(t, "adaptive", thinking["type"])
	assert.NotContains(t, thinking, "budget_tokens")
}

func TestHaiku5_InvokeSendsAdaptive(t *testing.T) {
	provider := &BedrockProvider{}
	for name, reasoning := range map[string]*schemas.ResponsesParametersReasoning{
		"effort":     {Effort: schemas.Ptr("high")},
		"max_tokens": {MaxTokens: schemas.Ptr(2048)},
	} {
		t.Run(name, func(t *testing.T) {
			ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
			req := &schemas.BifrostResponsesRequest{
				Provider: schemas.Bedrock,
				Model:    haiku5Model,
				Input: []schemas.ResponsesMessage{{
					Role:    schemas.Ptr(schemas.ResponsesInputMessageRoleUser),
					Content: &schemas.ResponsesMessageContent{ContentStr: schemas.Ptr("hi")},
				}},
				Params: &schemas.ResponsesParameters{MaxOutputTokens: schemas.Ptr(8192), Reasoning: reasoning},
			}
			body, bifrostErr := anthropic.BuildAnthropicResponsesRequestBody(ctx, req, provider.invokeBuildConfig(haiku5Model, false, true, nil))
			require.Nil(t, bifrostErr)
			assert.Equal(t, "adaptive", providerUtils.GetJSONField(body, "thinking.type").String(), "%s", body)
			assert.False(t, providerUtils.JSONFieldExists(body, "thinking.budget_tokens"), "%s", body)
		})
	}
}

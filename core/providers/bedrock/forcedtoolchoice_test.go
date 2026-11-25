package bedrock_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/maximhq/bifrost/core/providers/bedrock"
	"github.com/maximhq/bifrost/core/schemas"
)

// TestBedrockForcedToolChoiceDrop_Reported: Converse drops a forced choice for
// a model that rejects forced tool use (Fable 5.1+) on both builders and
// reports the drop as AdaptationToolChoice (BIT-3279).
func TestBedrockForcedToolChoiceDrop_Reported(t *testing.T) {
	const model = "us.anthropic.claude-fable-5-1"
	user := schemas.Ptr("hi")
	params := schemas.ToolFunctionParameters{Type: "object", Properties: schemas.NewOrderedMap()}

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	chat, err := bedrock.ToBedrockChatCompletionRequest(ctx, &schemas.BifrostChatRequest{
		Provider: schemas.Bedrock, Model: model,
		Input: []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: user}}},
		Params: &schemas.ChatParameters{
			Tools:      []schemas.ChatTool{{Type: schemas.ChatToolTypeFunction, Function: &schemas.ChatToolFunction{Name: "f", Parameters: &params}}},
			ToolChoice: &schemas.ChatToolChoice{ChatToolChoiceStr: schemas.Ptr("required")},
		},
	})
	require.NoError(t, err)
	require.NotNil(t, chat.ToolConfig)
	assert.Nil(t, chat.ToolConfig.ToolChoice, "chat choice dropped")
	assert.True(t, schemas.AdaptationsFromContext(ctx).Has(schemas.AdaptationToolChoice), "chat drop reported")

	ctx = schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	resp, err := bedrock.ToBedrockResponsesRequest(ctx, &schemas.BifrostResponsesRequest{
		Provider: schemas.Bedrock, Model: model,
		Input: []schemas.ResponsesMessage{{Role: schemas.Ptr(schemas.ResponsesInputMessageRoleUser), Content: &schemas.ResponsesMessageContent{ContentStr: user}}},
		Params: &schemas.ResponsesParameters{
			Tools: []schemas.ResponsesTool{{Type: schemas.ResponsesToolTypeFunction, Name: schemas.Ptr("f"),
				ResponsesToolFunction: &schemas.ResponsesToolFunction{Parameters: &params}}},
			ToolChoice: &schemas.ResponsesToolChoice{ResponsesToolChoiceStr: schemas.Ptr("required")},
		},
	})
	require.NoError(t, err)
	require.NotNil(t, resp.ToolConfig)
	assert.Nil(t, resp.ToolConfig.ToolChoice, "responses choice dropped")
	assert.True(t, schemas.AdaptationsFromContext(ctx).Has(schemas.AdaptationToolChoice), "responses drop reported")
}

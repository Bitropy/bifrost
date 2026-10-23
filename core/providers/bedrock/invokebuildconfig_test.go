package bedrock

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/maximhq/bifrost/core/providers/anthropic"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

// invokeBuildConfig is unexported, so its test lives in the internal package
// (it shipped in the external bedrock_test package, which does not compile).
func TestInvokeBuildConfig_ExcludesInternalConverseFields(t *testing.T) {
	provider := &BedrockProvider{}
	config := provider.invokeBuildConfig("us.anthropic.claude-sonnet-4-6", false, false, nil)

	excluded := make(map[string]bool)
	for _, f := range config.ExcludeFields {
		excluded[f] = true
	}

	assert.True(t, excluded["guardrailConfig"], "guardrailConfig must be excluded")
	assert.True(t, excluded["additionalModelRequestFieldPaths"], "additionalModelRequestFieldPaths must be excluded")
	assert.True(t, excluded["additionalModelResponseFieldPaths"], "additionalModelResponseFieldPaths must be excluded")
}

// A named function tool_choice with ParallelToolCalls=false reaches Claude
// exclusive on both Bedrock surfaces (BIT-2741 B3): the InvokeModel body
// (core's Anthropic converter) carries disable_parallel_tool_use:true on the
// {"type":"tool"} choice, and Converse carries it in
// additionalModelRequestFields. ParallelToolCalls is the caller's signal
// (bitrouter's toolchoice plugin sets it for an OpenAI named choice); without
// it the choice keeps Anthropic's semantics.
func TestNamedChoiceParallelToolCalls_BothSurfaces(t *testing.T) {
	req := func(parallel *bool) *schemas.BifrostChatRequest {
		return &schemas.BifrostChatRequest{
			Provider: schemas.Bedrock,
			Model:    "global.anthropic.claude-sonnet-4-6",
			Input:    []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("Time and weather?")}}},
			Params: &schemas.ChatParameters{
				Tools: parallelChatTools("get_weather", "get_time"),
				ToolChoice: &schemas.ChatToolChoice{ChatToolChoiceStruct: &schemas.ChatToolChoiceStruct{
					Type: schemas.ChatToolChoiceTypeFunction, Function: &schemas.ChatToolChoiceFunction{Name: "get_time"},
				}},
				ParallelToolCalls: parallel,
			},
		}
	}
	for _, tc := range []struct {
		name     string
		parallel *bool
		want     bool
	}{
		{"false", schemas.Ptr(false), true},
		{"unset", nil, false},
	} {
		t.Run("invoke/"+tc.name, func(t *testing.T) {
			ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
			provider := &BedrockProvider{}
			body, bifrostErr := anthropic.BuildAnthropicChatRequestBody(ctx, req(tc.parallel), provider.invokeBuildConfig("global.anthropic.claude-sonnet-4-6", false, false, nil))
			require.Nil(t, bifrostErr)
			assert.Equal(t, "tool", gjson.GetBytes(body, "tool_choice.type").String(), "%s", body)
			assert.Equal(t, "get_time", gjson.GetBytes(body, "tool_choice.name").String())
			assert.Equal(t, tc.want, gjson.GetBytes(body, "tool_choice.disable_parallel_tool_use").Bool(), "%s", body)
		})
		t.Run("converse/"+tc.name, func(t *testing.T) {
			ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
			out, err := ToBedrockChatCompletionRequest(ctx, req(tc.parallel))
			require.NoError(t, err)
			raw, err := json.Marshal(out)
			require.NoError(t, err)
			assert.Equal(t, tc.want, gjson.GetBytes(raw, "additionalModelRequestFields.tool_choice.disable_parallel_tool_use").Bool(), "%s", raw)
		})
	}
}

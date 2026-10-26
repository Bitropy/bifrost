package openai

import (
	"context"
	"strings"
	"testing"
	"time"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

// BIT-2764: the markers the Responses→Chat fallback now carries never reach
// the OpenAI chat wire: block, tool_call and tool markers are stripped as on
// any chat request, and the request-level marker stays in ExtraParams.
func TestResponsesChatFallback_CacheControlStrippedOnOpenAIWire(t *testing.T) {
	eph := &schemas.CacheControl{Type: schemas.CacheControlTypeEphemeral}
	rreq := &schemas.BifrostResponsesRequest{
		Provider: schemas.OpenAI, Model: "gpt-4.1",
		Input: []schemas.ResponsesMessage{
			{Type: schemas.Ptr(schemas.ResponsesMessageTypeMessage), Role: schemas.Ptr(schemas.ResponsesInputMessageRoleUser),
				Content: &schemas.ResponsesMessageContent{ContentBlocks: []schemas.ResponsesMessageContentBlock{
					{Type: schemas.ResponsesInputMessageContentBlockTypeText, Text: schemas.Ptr("q"), CacheControl: eph},
				}}},
			{Type: schemas.Ptr(schemas.ResponsesMessageTypeFunctionCall), CacheControl: eph,
				ResponsesToolMessage: &schemas.ResponsesToolMessage{CallID: schemas.Ptr("t1"), Name: schemas.Ptr("read"), Arguments: schemas.Ptr("{}")}},
			{Type: schemas.Ptr(schemas.ResponsesMessageTypeFunctionCallOutput), CacheControl: eph,
				ResponsesToolMessage: &schemas.ResponsesToolMessage{CallID: schemas.Ptr("t1"), Output: &schemas.ResponsesToolMessageOutputStruct{ResponsesToolCallOutputStr: schemas.Ptr("x")}}},
		},
		Params: &schemas.ResponsesParameters{
			Tools:       []schemas.ResponsesTool{{Type: schemas.ResponsesToolTypeFunction, Name: schemas.Ptr("read"), CacheControl: eph}},
			ExtraParams: map[string]interface{}{"cache_control": eph},
		},
	}
	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	defer ctx.Cancel()
	out, err := providerUtils.MarshalSorted(ToOpenAIChatRequest(ctx, rreq.ToChatRequest()))
	require.NoError(t, err)
	assert.NotContains(t, string(out), "cache_control", "%s", out)

	// The same request with no marker sends the same bytes: the marked text
	// block is still a string on this wire.
	unmarked := *rreq
	unmarked.Input = []schemas.ResponsesMessage{
		{Type: schemas.Ptr(schemas.ResponsesMessageTypeMessage), Role: schemas.Ptr(schemas.ResponsesInputMessageRoleUser),
			Content: &schemas.ResponsesMessageContent{ContentBlocks: []schemas.ResponsesMessageContentBlock{
				{Type: schemas.ResponsesInputMessageContentBlockTypeText, Text: schemas.Ptr("q")},
			}}},
		{Type: schemas.Ptr(schemas.ResponsesMessageTypeFunctionCall),
			ResponsesToolMessage: &schemas.ResponsesToolMessage{CallID: schemas.Ptr("t1"), Name: schemas.Ptr("read"), Arguments: schemas.Ptr("{}")}},
		{Type: schemas.Ptr(schemas.ResponsesMessageTypeFunctionCallOutput),
			ResponsesToolMessage: &schemas.ResponsesToolMessage{CallID: schemas.Ptr("t1"), Output: &schemas.ResponsesToolMessageOutputStruct{ResponsesToolCallOutputStr: schemas.Ptr("x")}}},
	}
	params := *rreq.Params
	params.Tools = []schemas.ResponsesTool{{Type: schemas.ResponsesToolTypeFunction, Name: schemas.Ptr("read")}}
	params.ExtraParams = nil
	unmarked.Params = &params
	want, err := providerUtils.MarshalSorted(ToOpenAIChatRequest(ctx, unmarked.ToChatRequest()))
	require.NoError(t, err)
	assert.JSONEq(t, string(want), string(out))
}

// A gpt-5.6 deployment served through the Responses API gets exactly one
// prompt_cache_breakpoint for a chat request from the Anthropic ingress: the
// tool result's (#6290). The other markers are not carried there (BIT-2764).
func TestChatToResponses_GPT56SingleToolResultBreakpoint(t *testing.T) {
	eph := &schemas.CacheControl{Type: schemas.CacheControlTypeEphemeral}
	chat := &schemas.BifrostChatRequest{
		Provider: schemas.OpenAI, Model: "gpt-5.6-terra",
		Input: []schemas.ChatMessage{
			{Role: schemas.ChatMessageRoleSystem, Content: &schemas.ChatMessageContent{ContentBlocks: []schemas.ChatContentBlock{{Type: schemas.ChatContentBlockTypeText, Text: schemas.Ptr("s"), CacheControl: eph}}}},
			{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{TextJoinOnWire: true, ContentBlocks: []schemas.ChatContentBlock{{Type: schemas.ChatContentBlockTypeText, Text: schemas.Ptr("q"), CacheControl: eph}}}},
			{Role: schemas.ChatMessageRoleAssistant, ChatAssistantMessage: &schemas.ChatAssistantMessage{ToolCalls: []schemas.ChatAssistantMessageToolCall{
				{ID: schemas.Ptr("t1"), Type: schemas.Ptr("function"), Function: schemas.ChatAssistantMessageToolCallFunction{Name: schemas.Ptr("read"), Arguments: "{}"}, CacheControl: eph},
			}}},
			{Role: schemas.ChatMessageRoleTool, ChatToolMessage: &schemas.ChatToolMessage{ToolCallID: schemas.Ptr("t1"), CacheControl: eph}, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("x")}},
		},
		Params: &schemas.ChatParameters{Tools: []schemas.ChatTool{{Type: schemas.ChatToolTypeFunction, Function: &schemas.ChatToolFunction{Name: "read"}, CacheControl: eph}}},
	}
	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	defer ctx.Cancel()
	out, err := providerUtils.MarshalSorted(ToOpenAIResponsesRequest(ctx, chat.ToResponsesRequest()))
	require.NoError(t, err)
	assert.Equal(t, 1, strings.Count(string(out), `"prompt_cache_breakpoint"`), "%s", out)
	assert.True(t, gjson.GetBytes(out, `input.#(type=="function_call_output").output.0.prompt_cache_breakpoint`).Exists(), "%s", out)
	assert.NotContains(t, string(out), "cache_control")
}

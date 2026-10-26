package bifrost

import (
	"context"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
)

// BIT-2764: a chat request served through the Responses API carries every
// cache_control to a Claude target (its Responses egress is the Anthropic
// wire) and only a tool result's to any other (the OpenAI Responses egress
// turns each carried marker into a prompt_cache_breakpoint).
func TestToResponsesRequestForTarget_CacheControlByTargetFamily(t *testing.T) {
	eph := &schemas.CacheControl{Type: schemas.CacheControlTypeEphemeral}
	chat := func(model string) *schemas.BifrostChatRequest {
		return &schemas.BifrostChatRequest{
			Provider: schemas.Azure, Model: model,
			Input: []schemas.ChatMessage{
				{Role: schemas.ChatMessageRoleSystem, Content: &schemas.ChatMessageContent{ContentBlocks: []schemas.ChatContentBlock{{Type: schemas.ChatContentBlockTypeText, Text: schemas.Ptr("s"), CacheControl: eph}}}},
				{Role: schemas.ChatMessageRoleAssistant, ChatAssistantMessage: &schemas.ChatAssistantMessage{ToolCalls: []schemas.ChatAssistantMessageToolCall{
					{ID: schemas.Ptr("t1"), Type: schemas.Ptr("function"), Function: schemas.ChatAssistantMessageToolCallFunction{Name: schemas.Ptr("read"), Arguments: "{}"}, CacheControl: eph},
				}}},
				{Role: schemas.ChatMessageRoleTool, ChatToolMessage: &schemas.ChatToolMessage{ToolCallID: schemas.Ptr("t1"), CacheControl: eph}, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("x")}},
			},
			Params: &schemas.ChatParameters{Tools: []schemas.ChatTool{{Type: schemas.ChatToolTypeFunction, Function: &schemas.ChatToolFunction{Name: "read"}, CacheControl: eph}}},
		}
	}
	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	defer ctx.Cancel()
	marks := func(r *schemas.BifrostResponsesRequest) (system, call, result, tool bool) {
		return r.Input[0].Content.ContentBlocks[0].CacheControl != nil, r.Input[1].CacheControl != nil,
			r.Input[2].CacheControl != nil, r.Params.Tools[0].CacheControl != nil
	}
	s, c, r, tl := marks(chat("claude-sonnet-4-5").ToResponsesRequestForTarget(ctx))
	assert.Equal(t, []bool{true, true, true, true}, []bool{s, c, r, tl}, "a Claude target gets every marker")
	s, c, r, tl = marks(chat("gpt-5.6-terra").ToResponsesRequestForTarget(ctx))
	assert.Equal(t, []bool{false, false, true, false}, []bool{s, c, r, tl}, "any other target only the tool result's")
}

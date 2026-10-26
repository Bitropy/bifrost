package schemas

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// BIT-2764: the Responses→Chat fallback carries cache_control. The Anthropic
// ingress converts a /v1/messages body to a Responses request first, so a
// marker the fallback drops never reaches an Anthropic-wire egress (Azure,
// Vertex and Bedrock Claude behind a chat route): every turn bills uncached.
func TestToChatRequest_CarriesCacheControl(t *testing.T) {
	eph := &CacheControl{Type: CacheControlTypeEphemeral}
	ttl := "1h"
	eph1h := &CacheControl{Type: CacheControlTypeEphemeral, TTL: &ttl}
	text := func(s string, cc *CacheControl) ResponsesMessageContentBlock {
		return ResponsesMessageContentBlock{Type: ResponsesInputMessageContentBlockTypeText, Text: Ptr(s), CacheControl: cc}
	}
	msg := func(role ResponsesMessageRoleType, blocks ...ResponsesMessageContentBlock) ResponsesMessage {
		return ResponsesMessage{Type: Ptr(ResponsesMessageTypeMessage), Role: Ptr(role), Content: &ResponsesMessageContent{ContentBlocks: blocks}}
	}
	outputBlocks := []ResponsesMessageContentBlock{text("B", nil)}
	req := &BifrostResponsesRequest{
		Input: []ResponsesMessage{
			msg(ResponsesInputMessageRoleSystem, text("sys one", nil), text("sys two", eph)),
			msg(ResponsesInputMessageRoleUser, text("plain", nil)),
			msg(ResponsesInputMessageRoleUser, text("marked", eph1h)),
			{Type: Ptr(ResponsesMessageTypeFunctionCall), ResponsesToolMessage: &ResponsesToolMessage{CallID: Ptr("t1"), Name: Ptr("read"), Arguments: Ptr("{}")}},
			{Type: Ptr(ResponsesMessageTypeFunctionCall), ResponsesToolMessage: &ResponsesToolMessage{CallID: Ptr("t2"), Name: Ptr("read"), Arguments: Ptr("{}")}, CacheControl: eph},
			{Type: Ptr(ResponsesMessageTypeFunctionCallOutput), ResponsesToolMessage: &ResponsesToolMessage{CallID: Ptr("t1"), Output: &ResponsesToolMessageOutputStruct{ResponsesToolCallOutputStr: Ptr("A")}}, CacheControl: eph},
			{Type: Ptr(ResponsesMessageTypeFunctionCallOutput), ResponsesToolMessage: &ResponsesToolMessage{CallID: Ptr("t2"), Output: &ResponsesToolMessageOutputStruct{ResponsesFunctionToolCallOutputBlocks: outputBlocks}}, CacheControl: eph},
			{Type: Ptr(ResponsesMessageTypeFunctionCallOutput), ResponsesToolMessage: &ResponsesToolMessage{CallID: Ptr("t3"), Output: &ResponsesToolMessageOutputStruct{ResponsesToolCallOutputStr: Ptr("")}}, CacheControl: eph},
		},
		Params: &ResponsesParameters{Tools: []ResponsesTool{
			{Type: ResponsesToolTypeFunction, Name: Ptr("read"), CacheControl: eph},
		}},
	}
	chat := req.ToChatRequest()
	in := chat.Input
	require.Len(t, in, 7)

	assert.Nil(t, in[0].Content.ContentStr)
	require.Len(t, in[0].Content.ContentBlocks, 2)
	assert.Nil(t, in[0].Content.ContentBlocks[0].CacheControl)
	assert.Same(t, eph, in[0].Content.ContentBlocks[1].CacheControl, "a system block's marker")

	require.NotNil(t, in[1].Content.ContentStr, "an unmarked single text block still collapses to a string")
	assert.Equal(t, "plain", *in[1].Content.ContentStr)

	assert.Nil(t, in[2].Content.ContentStr, "a marked single text block stays a block")
	require.Len(t, in[2].Content.ContentBlocks, 1)
	assert.Equal(t, ChatContentBlockTypeText, in[2].Content.ContentBlocks[0].Type)
	assert.Equal(t, "marked", *in[2].Content.ContentBlocks[0].Text)
	assert.Same(t, eph1h, in[2].Content.ContentBlocks[0].CacheControl, "ttl kept")
	assert.True(t, in[2].Content.TextJoinOnWire, "a string on every chat wire")
	wire, err := MarshalSorted(in[2].Content)
	require.NoError(t, err)
	assert.JSONEq(t, `"marked"`, string(wire))
	assert.False(t, in[0].Content.TextJoinOnWire, "a block list stays one")

	require.Len(t, in[3].ToolCalls, 2)
	assert.Nil(t, in[3].ToolCalls[0].CacheControl)
	assert.Same(t, eph, in[3].ToolCalls[1].CacheControl, "a tool_use's marker")

	for i, want := range map[int]string{4: `"A"`, 5: `"B"`, 6: `""`} {
		wire, err := MarshalSorted(in[i].Content)
		require.NoError(t, err)
		assert.JSONEq(t, want, string(wire), "turn %d: a tool result's content is unchanged", i)
		assert.Same(t, eph, in[i].ChatToolMessage.CacheControl, "turn %d: a tool_result's marker rides the tool message", i)
	}
	assert.Nil(t, outputBlocks[0].CacheControl, "the Responses item's blocks are not written")
	toolMsg, err := MarshalSorted(in[4])
	require.NoError(t, err)
	assert.NotContains(t, string(toolMsg), "cache_control", "the tool message's marker is off the chat wire")

	require.Len(t, chat.Params.Tools, 1)
	assert.Same(t, eph, chat.Params.Tools[0].CacheControl, "a tool's marker")
}

// A marker on a tool the fallback drops (a server tool, often the client's
// last, marking the whole tool list) moves onto the nearest kept tool before
// it; one with no kept tool before it has nowhere to go.
func TestSanitizeResponsesToolsForChatFallback_CarriesDroppedToolMarker(t *testing.T) {
	eph := &CacheControl{Type: CacheControlTypeEphemeral}
	fn := func(name string, cc *CacheControl) ResponsesTool {
		return ResponsesTool{Type: ResponsesToolTypeFunction, Name: Ptr(name), CacheControl: cc}
	}
	server := ResponsesTool{Type: "web_search_20250305", Name: Ptr("web_search"), CacheControl: eph}
	marks := func(tools []ChatTool) []bool {
		out := make([]bool, len(tools))
		for i := range tools {
			out[i] = tools[i].CacheControl != nil
		}
		return out
	}
	for _, tc := range []struct {
		name string
		in   []ResponsesTool
		want []bool
	}{
		{"own marker", []ResponsesTool{fn("a", nil), fn("b", eph)}, []bool{false, true}},
		{"dropped last tool's marker moves back", []ResponsesTool{fn("a", nil), fn("b", nil), server}, []bool{false, true}},
		{"kept tool already marked", []ResponsesTool{fn("a", eph), server}, []bool{true}},
		{"dropped first tool", []ResponsesTool{server, fn("a", nil)}, []bool{false}},
	} {
		assert.Equal(t, tc.want, marks(sanitizeResponsesToolsForChatFallback(tc.in)), tc.name)
	}
}

// Joined text blocks marshal as their newline-joined string; a non-text
// block keeps the list.
func TestChatMessageContent_TextJoinOnWire(t *testing.T) {
	eph := &CacheControl{Type: CacheControlTypeEphemeral}
	text := func(s string, cc *CacheControl) ChatContentBlock {
		return ChatContentBlock{Type: ChatContentBlockTypeText, Text: Ptr(s), CacheControl: cc}
	}
	for name, tc := range map[string]struct {
		c    ChatMessageContent
		want string
	}{
		"joined":        {ChatMessageContent{ContentBlocks: []ChatContentBlock{text("a", nil), text("b", eph)}, TextJoinOnWire: true}, `"a\nb"`},
		"not flagged":   {ChatMessageContent{ContentBlocks: []ChatContentBlock{text("a", nil)}}, `[{"text":"a","type":"text"}]`},
		"non-text kept": {ChatMessageContent{ContentBlocks: []ChatContentBlock{text("a", nil), {Type: ChatContentBlockTypeImage, ImageURLStruct: &ChatInputImage{URL: "u"}}}, TextJoinOnWire: true}, `[{"text":"a","type":"text"},{"image_url":{"url":"u"},"type":"image_url"}]`},
	} {
		got, err := MarshalSorted(tc.c)
		require.NoError(t, err, name)
		assert.JSONEq(t, tc.want, string(got), name)
	}
}

// The strip helpers copy only what they change and never write the input.
func TestChatMessagesWithoutCacheControl(t *testing.T) {
	eph := &CacheControl{Type: CacheControlTypeEphemeral}
	in := []ChatMessage{
		{Role: ChatMessageRoleUser, Content: &ChatMessageContent{ContentStr: Ptr("q")}},
		{Role: ChatMessageRoleSystem, Content: &ChatMessageContent{ContentBlocks: []ChatContentBlock{{Type: ChatContentBlockTypeText, Text: Ptr("s"), CacheControl: eph}}}},
		{Role: ChatMessageRoleAssistant, ChatAssistantMessage: &ChatAssistantMessage{ToolCalls: []ChatAssistantMessageToolCall{{ID: Ptr("t"), CacheControl: eph}}}},
		{Role: ChatMessageRoleTool, ChatToolMessage: &ChatToolMessage{ToolCallID: Ptr("t"), CacheControl: eph}, Content: &ChatMessageContent{ContentStr: Ptr("r")}},
	}
	out := ChatMessagesWithoutCacheControl(in)
	wire, err := MarshalSorted(out)
	require.NoError(t, err)
	assert.NotContains(t, string(wire), "cache_control")
	assert.Nil(t, out[3].ChatToolMessage.CacheControl)
	assert.Same(t, eph, in[1].Content.ContentBlocks[0].CacheControl, "the input is not written")
	assert.Same(t, eph, in[2].ToolCalls[0].CacheControl)
	assert.Same(t, eph, in[3].ChatToolMessage.CacheControl)
	assert.Same(t, in[0].Content, out[0].Content, "an unmarked message is not copied")
	plain := in[:1]
	assert.Equal(t, &plain[0], &ChatMessagesWithoutCacheControl(plain)[0], "nothing marked: the input itself")

	tools := []ChatTool{{Type: ChatToolTypeFunction, CacheControl: eph}}
	assert.Nil(t, ChatToolsWithoutCacheControl(tools)[0].CacheControl)
	assert.Same(t, eph, tools[0].CacheControl)
}

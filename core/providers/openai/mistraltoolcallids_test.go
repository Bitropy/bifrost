package openai

import (
	"regexp"
	"testing"

	"github.com/bytedance/sonic"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"

	"github.com/maximhq/bifrost/core/schemas"
)

var mistralToolCallIDRe = regexp.MustCompile(`^[a-zA-Z0-9]{9}$`)

// toolHistory is a replayed conversation: an OpenAI-style id, an
// Anthropic-style id, a valid nine-alphanumeric id (an earlier Mistral turn)
// and, twice, an id answered by two tool messages.
func toolHistory(provider schemas.ModelProvider, model string) *schemas.BifrostChatRequest {
	call := func(id, name string) schemas.ChatAssistantMessageToolCall {
		return schemas.ChatAssistantMessageToolCall{ID: schemas.Ptr(id), Type: schemas.Ptr("function"), Function: schemas.ChatAssistantMessageToolCallFunction{Name: schemas.Ptr(name), Arguments: "{}"}}
	}
	tool := func(id string) schemas.ChatMessage {
		return schemas.ChatMessage{Role: schemas.ChatMessageRoleTool, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("ok")}, ChatToolMessage: &schemas.ChatToolMessage{ToolCallID: schemas.Ptr(id)}}
	}
	return &schemas.BifrostChatRequest{
		Provider: provider,
		Model:    model,
		Input: []schemas.ChatMessage{
			{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("go")}},
			{Role: schemas.ChatMessageRoleAssistant, ChatAssistantMessage: &schemas.ChatAssistantMessage{ToolCalls: []schemas.ChatAssistantMessageToolCall{
				call("call_weather_1", "get_weather"), call("toolu_01A09q90qw90lq917835lq9", "get_time"),
			}}},
			tool("call_weather_1"),
			tool("toolu_01A09q90qw90lq917835lq9"),
			{Role: schemas.ChatMessageRoleAssistant, ChatAssistantMessage: &schemas.ChatAssistantMessage{ToolCalls: []schemas.ChatAssistantMessageToolCall{call("Ab3dE6gH9", "get_weather")}}},
			tool("Ab3dE6gH9"),
		},
	}
}

// BIT-2765: Mistral (native and the Vertex publisher) refuses any tool call
// id that is not ^[a-zA-Z0-9]{9}$ with a 400, so a history replayed from
// another model failed. The converter rewrites each such id deterministically,
// the same way on the assistant tool call and the tool message answering it,
// keeps a valid id, and never writes the caller's request.
func TestMistralToolCallIDs_Rewritten(t *testing.T) {
	for _, tc := range []struct {
		name     string
		provider schemas.ModelProvider
		model    string
	}{
		{"mistral", schemas.Mistral, "mistral-large-latest"},
		{"vertex-mistral", schemas.Vertex, "mistral-small-2503"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := toolHistory(tc.provider, tc.model)
			before, err := sonic.Marshal(req.Input)
			require.NoError(t, err)

			wire := marshalChat(t, ToOpenAIChatRequest(schemas.NewBifrostContext(nil, schemas.NoDeadline), req))

			after, err := sonic.Marshal(req.Input)
			require.NoError(t, err)
			assert.JSONEq(t, string(before), string(after), "the caller's request was written")

			ids := []string{
				gjson.Get(wire, "messages.1.tool_calls.0.id").String(),
				gjson.Get(wire, "messages.1.tool_calls.1.id").String(),
				gjson.Get(wire, "messages.4.tool_calls.0.id").String(),
			}
			for _, id := range ids {
				assert.Regexp(t, mistralToolCallIDRe, id, "wire: %s", wire)
			}
			assert.Equal(t, ids[0], gjson.Get(wire, "messages.2.tool_call_id").String(), "the tool message answers its call")
			assert.Equal(t, ids[1], gjson.Get(wire, "messages.3.tool_call_id").String(), "the tool message answers its call")
			assert.NotEqual(t, ids[0], ids[1])
			assert.Equal(t, "Ab3dE6gH9", ids[2], "a valid id is kept")
			assert.Equal(t, "Ab3dE6gH9", gjson.Get(wire, "messages.5.tool_call_id").String())

			again := marshalChat(t, ToOpenAIChatRequest(schemas.NewBifrostContext(nil, schemas.NoDeadline), toolHistory(tc.provider, tc.model)))
			assert.JSONEq(t, wire, again, "deterministic per request")
		})
	}
}

// A derived id never collides with an id already in the request: a valid id
// equal to the first derivation pushes the rewritten one to the next counter.
func TestMistralToolCallIDs_CollisionFree(t *testing.T) {
	taken := mistralToolCallID("call_weather_1", 0)
	req := &OpenAIChatRequest{Messages: []OpenAIMessage{
		{Role: schemas.ChatMessageRoleAssistant, OpenAIChatAssistantMessage: &OpenAIChatAssistantMessage{ToolCalls: []schemas.ChatAssistantMessageToolCall{
			{ID: schemas.Ptr("call_weather_1")}, {ID: schemas.Ptr(taken)},
		}}},
		{Role: schemas.ChatMessageRoleTool, ChatToolMessage: &schemas.ChatToolMessage{ToolCallID: schemas.Ptr("call_weather_1")}},
	}}
	req.rewriteMistralToolCallIDs()
	got := *req.Messages[0].OpenAIChatAssistantMessage.ToolCalls[0].ID
	assert.Equal(t, mistralToolCallID("call_weather_1", 1), got)
	assert.NotEqual(t, taken, got)
	assert.Equal(t, taken, *req.Messages[0].OpenAIChatAssistantMessage.ToolCalls[1].ID)
	assert.Equal(t, got, *req.Messages[1].ChatToolMessage.ToolCallID)
}

// Other wires are untouched: OpenAI accepts any id up to its own limit.
func TestMistralToolCallIDs_OnlyOnMistralWires(t *testing.T) {
	for _, tc := range []struct {
		provider schemas.ModelProvider
		model    string
	}{{schemas.OpenAI, "gpt-5"}, {schemas.Vertex, "meta/llama-4-maverick"}} {
		wire := marshalChat(t, ToOpenAIChatRequest(schemas.NewBifrostContext(nil, schemas.NoDeadline), toolHistory(tc.provider, tc.model)))
		assert.Equal(t, "call_weather_1", gjson.Get(wire, "messages.1.tool_calls.0.id").String(), "%s: %s", tc.provider, wire)
		assert.Equal(t, "call_weather_1", gjson.Get(wire, "messages.2.tool_call_id").String(), "%s: %s", tc.provider, wire)
	}
}

// A request without tool calls, or with only valid ids, allocates nothing.
func TestMistralToolCallIDs_NoOpAllocatesNothing(t *testing.T) {
	req := &OpenAIChatRequest{Messages: []OpenAIMessage{
		{Role: schemas.ChatMessageRoleUser},
		{Role: schemas.ChatMessageRoleAssistant, OpenAIChatAssistantMessage: &OpenAIChatAssistantMessage{ToolCalls: []schemas.ChatAssistantMessageToolCall{{ID: schemas.Ptr("Ab3dE6gH9")}}}},
		{Role: schemas.ChatMessageRoleTool, ChatToolMessage: &schemas.ChatToolMessage{ToolCallID: schemas.Ptr("Ab3dE6gH9")}},
	}}
	assert.Zero(t, testing.AllocsPerRun(100, req.rewriteMistralToolCallIDs))
}

func marshalChat(t *testing.T, req *OpenAIChatRequest) string {
	t.Helper()
	b, err := sonic.Marshal(req)
	require.NoError(t, err)
	return string(b)
}

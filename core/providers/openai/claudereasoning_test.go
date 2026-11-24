package openai

import (
	"context"
	"regexp"
	"testing"
	"time"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var mintedItemID = regexp.MustCompile(`"(fc_|rs_|msg_)[A-Za-z0-9]{50}"`)

// claudeToolTurn is a chat request from a Claude conversation with a tool turn
// whose assistant message carries reasoning (nil for none) and a marked text block.
func claudeToolTurn(model string, reasoning []schemas.ChatReasoningDetails) *schemas.BifrostChatRequest {
	eph := &schemas.CacheControl{Type: schemas.CacheControlTypeEphemeral}
	return &schemas.BifrostChatRequest{
		Provider: "my-openai", Model: model,
		Input: []schemas.ChatMessage{
			{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("q")}},
			{Role: schemas.ChatMessageRoleAssistant,
				Content: &schemas.ChatMessageContent{ContentBlocks: []schemas.ChatContentBlock{{Type: schemas.ChatContentBlockTypeText, Text: schemas.Ptr("reading"), CacheControl: eph}}},
				ChatAssistantMessage: &schemas.ChatAssistantMessage{
					ReasoningDetails: reasoning,
					ToolCalls: []schemas.ChatAssistantMessageToolCall{
						{ID: schemas.Ptr("t1"), Type: schemas.Ptr("function"), Function: schemas.ChatAssistantMessageToolCallFunction{Name: schemas.Ptr("read"), Arguments: "{}"}},
					}}},
			{Role: schemas.ChatMessageRoleTool, ChatToolMessage: &schemas.ChatToolMessage{ToolCallID: schemas.Ptr("t1")}, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("x")}},
		},
		Params: &schemas.ChatParameters{Tools: []schemas.ChatTool{{Type: schemas.ChatToolTypeFunction, Function: &schemas.ChatToolFunction{Name: "read"}}}},
	}
}

// openAIResponsesBody converts chat the way bitrouter's N1 attempt does (BIT-2764:
// cache_control carried for a Claude target) onto a custom OpenAI-base slug.
func openAIResponsesBody(t *testing.T, chat *schemas.BifrostChatRequest) (string, schemas.Adaptation) {
	t.Helper()
	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	t.Cleanup(ctx.Cancel)
	ctx.SetValue(schemas.BifrostContextKeyBaseProviderType, schemas.OpenAI)
	claude := schemas.IsAnthropicModelFamily(ctx, chat.Model)
	out, err := providerUtils.MarshalSorted(ToOpenAIResponsesRequest(ctx, chat.ToResponsesRequestFor(ctx, claude)))
	require.NoError(t, err)
	return mintedItemID.ReplaceAllString(string(out), `"<minted>"`), schemas.AdaptationsFromContext(ctx)
}

func signed(text string) schemas.ChatReasoningDetails {
	return schemas.ChatReasoningDetails{Type: schemas.BifrostReasoningDetailsTypeText, Text: schemas.Ptr(text), Signature: schemas.Ptr("sig")}
}

func redacted(data string) schemas.ChatReasoningDetails {
	return schemas.ChatReasoningDetails{Type: schemas.BifrostReasoningDetailsTypeEncrypted, Data: schemas.Ptr(data)}
}

// BIT-3177 S21: a Claude-named model on an OpenAI Responses egress cannot read
// Claude's signatures, so replayed thinking is removed and recorded as the
// reasoning adaptation. S22: the text block's cache_control is carried.
func TestClaudeOnOpenAIResponses_ReasoningRemovedIsAdapted(t *testing.T) {
	cases := []struct {
		name      string
		reasoning []schemas.ChatReasoningDetails
	}{
		{"thinking with text", []schemas.ChatReasoningDetails{signed("plan")}},
		{"signed thinking with no text", []schemas.ChatReasoningDetails{signed("")}},
		{"one redacted_thinking", []schemas.ChatReasoningDetails{redacted("enc")}},
		{"two redacted_thinking", []schemas.ChatReasoningDetails{redacted("e1"), redacted("e2")}},
		{"thinking and redacted_thinking", []schemas.ChatReasoningDetails{signed("plan"), redacted("enc")}},
	}
	plain, none := openAIResponsesBody(t, claudeToolTurn("claude-sonnet-4-5", nil))
	assert.Zero(t, none)
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			body, got := openAIResponsesBody(t, claudeToolTurn("claude-sonnet-4-5", tc.reasoning))
			assert.Equal(t, schemas.AdaptationReasoning, got)
			assert.NotContains(t, body, "sig")
			assert.NotContains(t, body, `"enc`)
			assert.NotContains(t, body, `"plan"`)
			// A content-only reasoning item is dropped; an encrypted one leaves an empty item.
			assert.Equal(t, plain, regexp.MustCompile(`\{"id":"<minted>","type":"reasoning","summary":\[\]\},`).ReplaceAllString(body, ""))
		})
	}
}

// With no Claude reasoning the body is the pre-BIT-3177 one and nothing is
// recorded. The OpenAI wire has no home for an assistant output_text block's
// cache_control, so S22's carried marker leaves these bytes unchanged.
func TestClaudeOnOpenAIResponses_NoReasoningGolden(t *testing.T) {
	const claudeGolden = `{"model":"claude-sonnet-4-5","input":[{"type":"message","role":"user","content":"q"},{"id":"<minted>","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"reading","annotations":[],"logprobs":[]}]},{"arguments":"{}","id":"<minted>","type":"function_call","call_id":"t1","name":"read"},{"type":"function_call_output","call_id":"t1","output":"x"}],"tools":[{"type":"function","name":"read"}]}`
	const gptGolden = `{"model":"gpt-4o","input":[{"type":"message","role":"user","content":"q"},{"id":"<minted>","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"reading","annotations":[],"logprobs":[]}]},{"arguments":"{}","id":"<minted>","type":"function_call","call_id":"t1","name":"read"},{"type":"function_call_output","call_id":"t1","output":"x"}],"tools":[{"type":"function","name":"read"}]}`
	body, got := openAIResponsesBody(t, claudeToolTurn("claude-sonnet-4-5", nil))
	assert.Zero(t, got)
	assert.Equal(t, claudeGolden, body)

	// A non-Claude target: the S22 carry does not apply and no bit is recorded,
	// even for a reasoning history the egress removes (S21 is the Claude route).
	body, got = openAIResponsesBody(t, claudeToolTurn("gpt-4o", nil))
	assert.Zero(t, got)
	assert.Equal(t, gptGolden, body)
	_, got = openAIResponsesBody(t, claudeToolTurn("gpt-4o", []schemas.ChatReasoningDetails{signed(""), redacted("e1"), redacted("e2")}))
	assert.Zero(t, got)
}

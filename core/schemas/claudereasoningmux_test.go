package schemas

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func assistantToolTurn(reasoning []ChatReasoningDetails, cc *CacheControl) *BifrostChatRequest {
	return &BifrostChatRequest{Model: "claude-sonnet-4-5", Input: []ChatMessage{{
		Role:    ChatMessageRoleAssistant,
		Content: &ChatMessageContent{ContentBlocks: []ChatContentBlock{{Type: ChatContentBlockTypeText, Text: Ptr("reading"), CacheControl: cc}}},
		ChatAssistantMessage: &ChatAssistantMessage{
			ReasoningDetails: reasoning,
			ToolCalls:        []ChatAssistantMessageToolCall{{ID: Ptr("t1"), Function: ChatAssistantMessageToolCallFunction{Name: Ptr("read"), Arguments: "{}"}}},
		},
	}}}
}

// BIT-3177 S21: the mux records the reasoning adaptation for a Claude target
// (carryCacheControl) when it leaves out a signed empty thinking block or a
// redacted_thinking block other than the last; what it emits is the egress's.
func TestToResponsesRequestFor_ClaudeReasoningDropIsAdapted(t *testing.T) {
	text := func(s string, sig *string) ChatReasoningDetails {
		return ChatReasoningDetails{Type: BifrostReasoningDetailsTypeText, Text: Ptr(s), Signature: sig}
	}
	enc := func(s string) ChatReasoningDetails {
		return ChatReasoningDetails{Type: BifrostReasoningDetailsTypeEncrypted, Data: Ptr(s)}
	}
	cases := []struct {
		name      string
		reasoning []ChatReasoningDetails
		claude    bool
		want      Adaptation
	}{
		{"signed thinking with no text", []ChatReasoningDetails{text("", Ptr("sig"))}, true, AdaptationReasoning},
		{"signed thinking with nil text", []ChatReasoningDetails{{Type: BifrostReasoningDetailsTypeText, Signature: Ptr("sig")}}, true, AdaptationReasoning},
		{"two redacted_thinking", []ChatReasoningDetails{enc("e1"), enc("e2")}, true, AdaptationReasoning},
		{"signed thinking with text and one redacted_thinking: carried", []ChatReasoningDetails{text("plan", Ptr("sig")), enc("e1")}, true, 0},
		{"unsigned empty thinking: nothing to lose", []ChatReasoningDetails{text("", nil)}, true, 0},
		{"no reasoning", nil, true, 0},
		{"non-Claude target", []ChatReasoningDetails{text("", Ptr("sig")), enc("e1"), enc("e2")}, false, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := NewBifrostContext(context.Background(), time.Time{})
			t.Cleanup(ctx.Cancel)
			assistantToolTurn(tc.reasoning, nil).ToResponsesRequestFor(ctx, tc.claude)
			assert.Equal(t, tc.want, AdaptationsFromContext(ctx))
		})
	}
}

// BIT-3177 S22: the text sent alongside tool calls keeps its cache_control on a
// Claude target, like the turn's other markers; any other target carries none.
func TestToResponsesRequestFor_ToolTurnTextKeepsCacheControl(t *testing.T) {
	eph := &CacheControl{Type: CacheControlTypeEphemeral}
	for _, claude := range []bool{true, false} {
		in := assistantToolTurn(nil, eph).ToResponsesRequestFor(nil, claude).Input
		require.Len(t, in, 2)
		require.Equal(t, ResponsesMessageTypeMessage, *in[0].Type)
		require.Len(t, in[0].Content.ContentBlocks, 1)
		if claude {
			assert.Same(t, eph, in[0].Content.ContentBlocks[0].CacheControl)
		} else {
			assert.Nil(t, in[0].Content.ContentBlocks[0].CacheControl)
		}
	}
}

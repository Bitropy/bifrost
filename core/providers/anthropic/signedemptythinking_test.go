package anthropic

import (
	"context"
	"testing"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/tidwall/gjson"
)

func TestReplayedThinkingText(t *testing.T) {
	empty := ""
	text := "t"
	sig := "S"
	tests := []struct {
		name      string
		text, sig *string
		want      *string
	}{
		{"no text, no signature", nil, nil, nil},
		{"no text, empty signature", nil, &empty, nil},
		{"no text, signed", nil, &sig, &empty},
		{"empty text, signed", &empty, &sig, &empty},
		{"text, unsigned", &text, nil, &text},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := replayedThinkingText(tt.text, tt.sig)
			switch {
			case tt.want == nil && got != nil:
				t.Fatalf("got %q, want nil", *got)
			case tt.want != nil && (got == nil || *got != *tt.want):
				t.Fatalf("got %v, want %q", got, *tt.want)
			}
			if tt.text != nil && got != tt.text {
				t.Error("text was not returned as given")
			}
		})
	}
}

// With an OpenAI source the item ID is embedded into the signature; an unsigned
// block must not count as signed because of that embedding.
func TestConvertBifrostReasoning_EmbeddedIDIsNotASignature(t *testing.T) {
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	t.Cleanup(ctx.Cancel)
	if !providerUtils.ShouldEmbedReasoningItemID(ctx, schemas.OpenAI, "gpt-5") {
		t.Fatal("fixture: an OpenAI source must embed the item ID")
	}
	msg := &schemas.ResponsesMessage{
		ID:      schemas.Ptr("rs_1"),
		Type:    schemas.Ptr(schemas.ResponsesMessageTypeReasoning),
		Content: &schemas.ResponsesMessageContent{ContentBlocks: []schemas.ResponsesMessageContentBlock{{Type: schemas.ResponsesOutputMessageContentTypeReasoning}}},
	}
	if got := convertBifrostReasoningToAnthropicThinking(ctx, msg, schemas.OpenAI, "gpt-5", false); len(got) != 0 {
		t.Fatalf("got %d blocks, want none: %+v", len(got), got)
	}
}

func signedTextDetail(index int, text *string, sig string) schemas.ChatReasoningDetails {
	return schemas.ChatReasoningDetails{Index: index, Type: schemas.BifrostReasoningDetailsTypeText, Text: text, Signature: schemas.Ptr(sig)}
}

func signedThinkingChatRequest(provider schemas.ModelProvider, details []schemas.ChatReasoningDetails) *schemas.BifrostChatRequest {
	return &schemas.BifrostChatRequest{
		Provider: provider,
		Model:    "claude-haiku-5-5",
		Input: []schemas.ChatMessage{
			{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("find it")}},
			{
				Role: schemas.ChatMessageRoleAssistant,
				ChatAssistantMessage: &schemas.ChatAssistantMessage{
					ReasoningDetails: details,
					ToolCalls: []schemas.ChatAssistantMessageToolCall{{
						ID: schemas.Ptr("toolu_1"), Type: schemas.Ptr("function"),
						Function: schemas.ChatAssistantMessageToolCallFunction{Name: schemas.Ptr("f"), Arguments: `{"a":1}`},
					}},
				},
			},
			{Role: schemas.ChatMessageRoleTool, ChatToolMessage: &schemas.ChatToolMessage{ToolCallID: schemas.Ptr("toolu_1")},
				Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("ok")}},
		},
		Params: &schemas.ChatParameters{MaxCompletionTokens: schemas.Ptr(256)},
	}
}

func buildChatBody(t *testing.T, req *schemas.BifrostChatRequest) []byte {
	t.Helper()
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	t.Cleanup(ctx.Cancel)
	body, bErr := BuildAnthropicChatRequestBody(ctx, req, AnthropicRequestBuildConfig{Provider: req.Provider})
	if bErr != nil {
		t.Fatalf("build: %+v", bErr.Error)
	}
	return body
}

func buildResponsesBody(t *testing.T, req *schemas.BifrostResponsesRequest) []byte {
	t.Helper()
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	t.Cleanup(ctx.Cancel)
	body, bErr := BuildAnthropicResponsesRequestBody(ctx, req, AnthropicRequestBuildConfig{Provider: req.Provider})
	if bErr != nil {
		t.Fatalf("build: %+v", bErr.Error)
	}
	return body
}

// assistantBlocks returns the raw content blocks of the first assistant message.
func assistantBlocks(t *testing.T, body []byte) []gjson.Result {
	t.Helper()
	msg := gjson.GetBytes(body, `messages.#(role=="assistant")`)
	if !msg.Exists() {
		t.Fatalf("no assistant message: %s", body)
	}
	return msg.Get("content").Array()
}

func wantBlocks(t *testing.T, body []byte, want ...string) {
	t.Helper()
	got := assistantBlocks(t, body)
	if len(got) != len(want) {
		t.Fatalf("got %d blocks, want %d: %s", len(got), len(want), body)
	}
	for i, w := range want {
		if w == "" {
			continue
		}
		if got[i].Raw != w {
			t.Errorf("block %d\n got  %s\n want %s", i, got[i].Raw, w)
		}
	}
}

func TestBuildAnthropicChatRequestBody_KeepsSignedEmptyThinking(t *testing.T) {
	const signed = `{"type":"thinking","thinking":"","signature":"S"}`
	shapes := map[string]schemas.ChatReasoningDetails{
		"unary shape":  signedTextDetail(0, schemas.Ptr(""), "S"),
		"stream shape": signedTextDetail(0, nil, "S"),
	}
	for _, provider := range []schemas.ModelProvider{schemas.Anthropic, schemas.Vertex} {
		for name, detail := range shapes {
			t.Run(string(provider)+"/"+name, func(t *testing.T) {
				body := buildChatBody(t, signedThinkingChatRequest(provider, []schemas.ChatReasoningDetails{detail}))
				wantBlocks(t, body, signed, "")
				if typ := assistantBlocks(t, body)[1].Get("type").String(); typ != "tool_use" {
					t.Errorf("block 1 is %q, want tool_use", typ)
				}
			})
		}
		t.Run(string(provider)+"/encrypted without data", func(t *testing.T) {
			gemini := schemas.ChatReasoningDetails{Type: schemas.BifrostReasoningDetailsTypeEncrypted, Signature: schemas.Ptr("G")}
			body := buildChatBody(t, signedThinkingChatRequest(provider, []schemas.ChatReasoningDetails{gemini}))
			wantBlocks(t, body, "")
		})
	}
}

// Under summarized display one turn holds signed empty blocks next to signed
// blocks with text; dropping only some of them is a 400.
func TestBuildAnthropicRequestBody_KeepsMixedSignedTurn(t *testing.T) {
	details := []schemas.ChatReasoningDetails{
		signedTextDetail(0, schemas.Ptr(""), "S1"),
		signedTextDetail(1, schemas.Ptr("progress"), "S2"),
		signedTextDetail(2, schemas.Ptr(""), "S3"),
	}
	want := []string{
		`{"type":"thinking","thinking":"","signature":"S1"}`,
		`{"type":"thinking","thinking":"progress","signature":"S2"}`,
		`{"type":"thinking","thinking":"","signature":"S3"}`,
		"",
	}
	t.Run("chat", func(t *testing.T) {
		body := buildChatBody(t, signedThinkingChatRequest(schemas.Anthropic, details))
		wantBlocks(t, body, want...)
		if typ := assistantBlocks(t, body)[3].Get("type").String(); typ != "tool_use" {
			t.Errorf("block 3 is %q, want tool_use", typ)
		}
	})
	t.Run("responses", func(t *testing.T) {
		blocks := make([]schemas.ResponsesMessageContentBlock, len(details))
		for i, d := range details {
			blocks[i] = schemas.ResponsesMessageContentBlock{Type: schemas.ResponsesOutputMessageContentTypeReasoning, Text: d.Text, Signature: d.Signature}
		}
		req := responsesReplay(reasoningItem(nil, blocks, nil))
		req.Input[2] = schemas.ResponsesMessage{
			Type: schemas.Ptr(schemas.ResponsesMessageTypeFunctionCall),
			ResponsesToolMessage: &schemas.ResponsesToolMessage{
				CallID: schemas.Ptr("toolu_1"), Name: schemas.Ptr("f"), Arguments: schemas.Ptr(`{"a":1}`),
			},
		}
		body := buildResponsesBody(t, req)
		wantBlocks(t, body, want...)
		if typ := assistantBlocks(t, body)[3].Get("type").String(); typ != "tool_use" {
			t.Errorf("block 3 is %q, want tool_use", typ)
		}
	})
}

func reasoningItem(id *string, blocks []schemas.ResponsesMessageContentBlock, reasoning *schemas.ResponsesReasoning) schemas.ResponsesMessage {
	msg := schemas.ResponsesMessage{
		ID:                 id,
		Type:               schemas.Ptr(schemas.ResponsesMessageTypeReasoning),
		Role:               schemas.Ptr(schemas.ResponsesInputMessageRoleAssistant),
		ResponsesReasoning: reasoning,
	}
	if blocks != nil {
		msg.Content = &schemas.ResponsesMessageContent{ContentBlocks: blocks}
	}
	return msg
}

func responsesReplay(item schemas.ResponsesMessage) *schemas.BifrostResponsesRequest {
	return &schemas.BifrostResponsesRequest{
		Provider: schemas.Anthropic,
		Model:    "claude-haiku-5-5",
		Input: []schemas.ResponsesMessage{
			{Role: schemas.Ptr(schemas.ResponsesInputMessageRoleUser), Type: schemas.Ptr(schemas.ResponsesMessageTypeMessage),
				Content: &schemas.ResponsesMessageContent{ContentStr: schemas.Ptr("find it")}},
			item,
			{Type: schemas.Ptr(schemas.ResponsesMessageTypeMessage), Role: schemas.Ptr(schemas.ResponsesInputMessageRoleAssistant),
				Content: &schemas.ResponsesMessageContent{ContentStr: schemas.Ptr("found")}},
		},
		Params: &schemas.ResponsesParameters{MaxOutputTokens: schemas.Ptr(256)},
	}
}

func TestBuildAnthropicResponsesRequestBody_KeepsSignedEmptyThinking(t *testing.T) {
	const signed = `{"type":"thinking","thinking":"","signature":"S"}`
	reasoningBlock := func(text, sig *string) []schemas.ResponsesMessageContentBlock {
		return []schemas.ResponsesMessageContentBlock{{Type: schemas.ResponsesOutputMessageContentTypeReasoning, Text: text, Signature: sig}}
	}
	encrypted := &schemas.ResponsesReasoning{Summary: []schemas.ResponsesReasoningSummary{}, EncryptedContent: schemas.Ptr("ENC")}
	tests := []struct {
		name string
		item schemas.ResponsesMessage
		want []string
	}{
		{"empty text, signed", reasoningItem(nil, reasoningBlock(schemas.Ptr(""), schemas.Ptr("S")), nil), []string{signed, ""}},
		{"no text, signed", reasoningItem(nil, reasoningBlock(nil, schemas.Ptr("S")), nil), []string{signed, ""}},
		{"no text, unsigned", reasoningItem(nil, reasoningBlock(nil, nil), nil), []string{""}},
		{"item ID, empty text, unsigned", reasoningItem(schemas.Ptr("rs_1"), reasoningBlock(schemas.Ptr(""), nil), nil), []string{""}},
		{"item ID, empty summary only", reasoningItem(schemas.Ptr("rs_1"), nil,
			&schemas.ResponsesReasoning{Summary: []schemas.ResponsesReasoningSummary{{Type: schemas.ResponsesReasoningContentBlockTypeSummaryText, Text: ""}}}), []string{""}},
		{"no text, signed, encrypted_content", reasoningItem(nil, reasoningBlock(nil, schemas.Ptr("S")), encrypted), []string{signed, ""}},
		{"empty text, signed, encrypted_content", reasoningItem(nil, reasoningBlock(schemas.Ptr(""), schemas.Ptr("S")), encrypted), []string{signed, ""}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wantBlocks(t, buildResponsesBody(t, responsesReplay(tt.item)), tt.want...)
		})
	}
}

package bifrost

import (
	"context"
	"regexp"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/internal/wirecapture"
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

// fcItemID matches the function_call item id core mints per request; it is not under test.
var fcItemID = regexp.MustCompile(`"id":"fc_[0-9a-f]+"`)

// BIT-2906: a chat request served through the Responses API (BifrostContextKeyChangeRequestType)
// carries in a function_call_output only the media every Responses provider reads; the rest is a
// placeholder: without a size under a request-body rewriter, upstream's sized text without one.
func TestChatThroughResponses_ToolResultMediaPlaceholderOmitsSizeUnderRewriter(t *testing.T) {
	server := wirecapture.NewServer(t)
	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 4, server.URL)
	account.configs[schemas.OpenAI].NetworkConfig.MaxRetries = 0
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{{ID: "k", Value: *schemas.NewSecretVar("sk-a"), Models: schemas.WhiteList{"*"}, Weight: 1}})
	client, err := Init(context.Background(), schemas.BifrostConfig{Account: account, Logger: NewDefaultLogger(schemas.LogLevelError)})
	if err != nil {
		t.Fatalf("Init: %v", err)
	}
	t.Cleanup(client.Shutdown)
	request := func() *schemas.BifrostChatRequest {
		return &schemas.BifrostChatRequest{Provider: schemas.OpenAI, Model: "gpt-4o", Input: wirecapture.ChatToolResult(schemas.ChatContentBlock{
			Type: schemas.ChatContentBlockTypeInputAudio, InputAudio: &schemas.ChatInputAudio{Data: "AAAAAAAAAAAA", Format: schemas.Ptr("wav")},
		})}
	}
	for _, row := range []struct {
		name string
		send func(ctx *schemas.BifrostContext)
	}{
		{"unary", func(ctx *schemas.BifrostContext) { _, _ = client.ChatCompletionRequest(ctx, request()) }},
		{"stream", func(ctx *schemas.BifrostContext) {
			chunks, _ := client.ChatCompletionStreamRequest(ctx, request())
			wirecapture.Drain(chunks)
		}},
	} {
		t.Run(row.name, func(t *testing.T) {
			wirecapture.AssertPlaceholderSizeFreeUnderRewriter(t, "[audio/wav, 9 bytes returned by the tool, not shown]", "[audio/wav returned by the tool, not shown]", func(ctx *schemas.BifrostContext) []byte {
				ctx.SetValue(schemas.BifrostContextKeyChangeRequestType, schemas.ResponsesRequest)
				row.send(ctx)
				return fcItemID.ReplaceAll(server.Next(t), []byte(`"id":"fc_"`))
			})
		})
	}
}

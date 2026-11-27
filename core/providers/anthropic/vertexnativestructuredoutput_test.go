package anthropic

import (
	"strings"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The tests below install a process-global capability resolver, so none of them runs in parallel.

func soChatRequest(provider schemas.ModelProvider, model string, effort *string) *schemas.BifrostChatRequest {
	rf := makeSOResponseFormat("my_schema")
	params := &schemas.ChatParameters{ResponseFormat: &rf}
	if effort != nil {
		params.MaxCompletionTokens = new(16000)
		params.Reasoning = &schemas.ChatReasoning{Effort: effort}
	}
	return &schemas.BifrostChatRequest{
		Provider: provider,
		Model:    model,
		Input: []schemas.ChatMessage{
			{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("Hello")}},
		},
		Params: params,
	}
}

func soResponsesRequest(provider schemas.ModelProvider, model string, blocks []schemas.ResponsesMessageContentBlock) *schemas.BifrostResponsesRequest {
	content := &schemas.ResponsesMessageContent{ContentStr: schemas.Ptr("Hello")}
	if blocks != nil {
		content = &schemas.ResponsesMessageContent{ContentBlocks: blocks}
	}
	return &schemas.BifrostResponsesRequest{
		Provider: provider,
		Model:    model,
		Input:    []schemas.ResponsesMessage{{Role: schemas.Ptr(schemas.ResponsesInputMessageRoleUser), Content: content}},
		Params:   &schemas.ResponsesParameters{Text: makeResponsesTextFormat("my_schema")},
	}
}

func standInTool(req *AnthropicMessageRequest) *AnthropicTool {
	for i := range req.Tools {
		if strings.HasPrefix(req.Tools[i].Name, "bf_so_") {
			return &req.Tools[i]
		}
	}
	return nil
}

func toolChoiceAdapted(ctx *schemas.BifrostContext) bool {
	return schemas.AdaptationsFromContext(ctx)&schemas.AdaptationToolChoice != 0
}

func assertNativeFormat(t *testing.T, ctx *schemas.BifrostContext, req *AnthropicMessageRequest) {
	t.Helper()
	require.NotNil(t, req.OutputConfig, "output_config must carry the schema")
	assert.Contains(t, string(req.OutputConfig.Format), `"json_schema"`)
	assert.Nil(t, standInTool(req), "no stand-in tool beside the native format")
	assert.Nil(t, req.ToolChoice)
	assert.False(t, toolChoiceAdapted(ctx), "nothing is dropped, so no tool_choice adaptation")
}

func assertForcedStandIn(t *testing.T, ctx *schemas.BifrostContext, req *AnthropicMessageRequest) {
	t.Helper()
	tool := standInTool(req)
	require.NotNil(t, tool, "the stand-in tool must be sent")
	require.NotNil(t, req.ToolChoice)
	assert.Equal(t, "tool", req.ToolChoice.Type)
	assert.Equal(t, tool.Name, req.ToolChoice.Name)
	assert.True(t, req.OutputConfig == nil || req.OutputConfig.Format == nil, "no output_config.format")
	assert.False(t, toolChoiceAdapted(ctx))
}

func assertUnforcedStandIn(t *testing.T, ctx *schemas.BifrostContext, req *AnthropicMessageRequest) {
	t.Helper()
	require.NotNil(t, standInTool(req), "the stand-in tool must be sent")
	assert.Nil(t, req.ToolChoice)
	assert.True(t, req.OutputConfig == nil || req.OutputConfig.Format == nil, "no output_config.format")
	assert.True(t, toolChoiceAdapted(ctx), "the dropped pin is recorded")
}

func noForcedToolChoice() schemas.ModelCapabilities {
	return schemas.ModelCapabilities{SupportsForcedToolChoice: new(false)}
}

func TestVertexNativeStructuredOutput_Chat(t *testing.T) {
	t.Run("AC-1 record says forced tool_choice is unsupported", func(t *testing.T) {
		setProviderOverride(t, schemas.Vertex, "claude-sonnet-5-5", noForcedToolChoice())
		ctx := schemas.NewBifrostContext(nil, schemas.NoDeadline)
		out, err := ToAnthropicChatRequest(ctx, soChatRequest(schemas.Vertex, "claude-sonnet-5-5", nil))
		require.NoError(t, err)
		assertNativeFormat(t, ctx, out)
	})

	t.Run("AC-2 unforceable by name default", func(t *testing.T) {
		ctx := schemas.NewBifrostContext(nil, schemas.NoDeadline)
		out, err := ToAnthropicChatRequest(ctx, soChatRequest(schemas.Vertex, "claude-opus-5-5", nil))
		require.NoError(t, err)
		assertNativeFormat(t, ctx, out)
	})

	t.Run("AC-3 thinking on keeps effort beside format", func(t *testing.T) {
		ctx := schemas.NewBifrostContext(nil, schemas.NoDeadline)
		out, err := ToAnthropicChatRequest(ctx, soChatRequest(schemas.Vertex, "claude-sonnet-4-6", schemas.Ptr("high")))
		require.NoError(t, err)
		assertNativeFormat(t, ctx, out)
		require.NotNil(t, out.OutputConfig.Effort)
		assert.Equal(t, "high", *out.OutputConfig.Effort)
	})

	t.Run("AC-4 forceable and thinking off keeps the forced stand-in", func(t *testing.T) {
		ctx := schemas.NewBifrostContext(nil, schemas.NoDeadline)
		out, err := ToAnthropicChatRequest(ctx, soChatRequest(schemas.Vertex, "claude-sonnet-4-5@20250929", nil))
		require.NoError(t, err)
		assertForcedStandIn(t, ctx, out)
	})

	t.Run("AC-5 pre-4.5 under thinking keeps the unforced stand-in", func(t *testing.T) {
		ctx := schemas.NewBifrostContext(nil, schemas.NoDeadline)
		out, err := ToAnthropicChatRequest(ctx, soChatRequest(schemas.Vertex, "claude-sonnet-4@20250514", schemas.Ptr("high")))
		require.NoError(t, err)
		assertUnforcedStandIn(t, ctx, out)
	})

	t.Run("AC-6 record says native structured output is unsupported", func(t *testing.T) {
		setProviderOverride(t, schemas.Vertex, "claude-sonnet-5-5", schemas.ModelCapabilities{
			SupportsForcedToolChoice:       new(false),
			SupportsNativeStructuredOutput: new(false),
		})
		ctx := schemas.NewBifrostContext(nil, schemas.NoDeadline)
		out, err := ToAnthropicChatRequest(ctx, soChatRequest(schemas.Vertex, "claude-sonnet-5-5", nil))
		require.NoError(t, err)
		assertUnforcedStandIn(t, ctx, out)
	})
}

// TestVertexNativeStructuredOutput_OtherSyntheticProviders is AC-7: only Vertex goes native.
func TestVertexNativeStructuredOutput_OtherSyntheticProviders(t *testing.T) {
	for _, provider := range []schemas.ModelProvider{schemas.Azure, schemas.BedrockMantle} {
		t.Run(string(provider)+"/forced unsupported", func(t *testing.T) {
			setProviderOverride(t, provider, "claude-sonnet-5-5", noForcedToolChoice())
			ctx := schemas.NewBifrostContext(nil, schemas.NoDeadline)
			out, err := ToAnthropicChatRequest(ctx, soChatRequest(provider, "claude-sonnet-5-5", nil))
			require.NoError(t, err)
			assertUnforcedStandIn(t, ctx, out)
		})
		t.Run(string(provider)+"/thinking on", func(t *testing.T) {
			ctx := schemas.NewBifrostContext(nil, schemas.NoDeadline)
			out, err := ToAnthropicChatRequest(ctx, soChatRequest(provider, "claude-sonnet-4-6", schemas.Ptr("high")))
			require.NoError(t, err)
			assertUnforcedStandIn(t, ctx, out)
		})
	}
}

// TestStructuredOutput_KeepsTaskBudget is AC-8: the format joins an output_config another field wrote.
func TestStructuredOutput_KeepsTaskBudget(t *testing.T) {
	req := soChatRequest(schemas.Anthropic, "claude-opus-4-7", nil)
	req.Params.TaskBudget = &schemas.ChatTaskBudget{Type: "tokens", Total: 20000}
	ctx := schemas.NewBifrostContext(nil, schemas.NoDeadline)
	out, err := ToAnthropicChatRequest(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, out.OutputConfig)
	assert.Contains(t, string(out.OutputConfig.Format), `"json_schema"`)
	require.NotNil(t, out.OutputConfig.TaskBudget)
	assert.Equal(t, 20000, out.OutputConfig.TaskBudget.Total)
}

// TestIsClaude45Plus is AC-9.
func TestIsClaude45Plus(t *testing.T) {
	for model, want := range map[string]bool{
		"claude-sonnet-4-5@20250929": true,
		"claude-haiku-4-5@20251001":  true,
		"claude-opus-4-8":            true,
		"claude-sonnet-5-5":          true,
		"claude-fable-5-1":           true,
		"claude-opus-4-1@20250805":   false,
		"claude-sonnet-4@20250514":   false,
		"claude-3-5-haiku":           false,
	} {
		assert.Equal(t, want, IsClaude45Plus(model), model)
	}
}

// TestVertexNativeStructuredOutput_Responses is AC-10 and AC-11.
func TestVertexNativeStructuredOutput_Responses(t *testing.T) {
	t.Run("AC-10 AC-1 on Responses", func(t *testing.T) {
		setProviderOverride(t, schemas.Vertex, "claude-sonnet-5-5", noForcedToolChoice())
		ctx := schemas.NewBifrostContext(nil, schemas.NoDeadline)
		out, err := ToAnthropicResponsesRequest(ctx, soResponsesRequest(schemas.Vertex, "claude-sonnet-5-5", nil))
		require.NoError(t, err)
		assertNativeFormat(t, ctx, out)
	})

	t.Run("AC-10 AC-4 on Responses", func(t *testing.T) {
		ctx := schemas.NewBifrostContext(nil, schemas.NoDeadline)
		out, err := ToAnthropicResponsesRequest(ctx, soResponsesRequest(schemas.Vertex, "claude-sonnet-4-5@20250929", nil))
		require.NoError(t, err)
		assertForcedStandIn(t, ctx, out)
	})

	t.Run("AC-11 citations keep the unforced stand-in", func(t *testing.T) {
		setProviderOverride(t, schemas.Vertex, "claude-sonnet-5-5", noForcedToolChoice())
		blocks := []schemas.ResponsesMessageContentBlock{{
			Type: schemas.ResponsesInputMessageContentBlockTypeFile,
			ResponsesInputMessageContentBlockFile: &schemas.ResponsesInputMessageContentBlockFile{
				FileData: schemas.Ptr("plain text document"),
				FileType: schemas.Ptr("text/plain"),
			},
			Citations: &schemas.Citations{Enabled: schemas.Ptr(true)},
		}}
		ctx := schemas.NewBifrostContext(nil, schemas.NoDeadline)
		out, err := ToAnthropicResponsesRequest(ctx, soResponsesRequest(schemas.Vertex, "claude-sonnet-5-5", blocks))
		require.NoError(t, err)
		assertUnforcedStandIn(t, ctx, out)
	})
}

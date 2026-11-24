package bifrost

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/maximhq/bifrost/core/providers/anthropic"
	"github.com/maximhq/bifrost/core/providers/bedrock"
	"github.com/maximhq/bifrost/core/providers/gemini"
	"github.com/maximhq/bifrost/core/providers/openai"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/tidwall/gjson"
)

// emptyCombinatorWire is one wire: the provider and model whose attempt is normalized,
// the converter that builds its request, and where the tool schema sits in the body.
type emptyCombinatorWire struct {
	name      string
	provider  schemas.ModelProvider
	model     string
	responses bool
	path      string
	build     func(ctx *schemas.BifrostContext, chat *schemas.BifrostChatRequest, resp *schemas.BifrostResponsesRequest) (any, error)
}

var emptyCombinatorWires = []emptyCombinatorWire{
	{name: "openai_chat", provider: schemas.OpenAI, model: "gpt-5-mini", path: "tools.0.function.parameters",
		build: func(ctx *schemas.BifrostContext, c *schemas.BifrostChatRequest, _ *schemas.BifrostResponsesRequest) (any, error) {
			return openai.ToOpenAIChatRequest(ctx, c), nil
		}},
	{name: "openai_responses", provider: schemas.OpenAI, model: "gpt-5-mini", responses: true, path: "tools.0.parameters",
		build: func(ctx *schemas.BifrostContext, _ *schemas.BifrostChatRequest, r *schemas.BifrostResponsesRequest) (any, error) {
			return openai.ToOpenAIResponsesRequest(ctx, r), nil
		}},
	{name: "anthropic_chat", provider: schemas.Anthropic, model: "claude-sonnet-4-5", path: "tools.0.input_schema",
		build: func(ctx *schemas.BifrostContext, c *schemas.BifrostChatRequest, _ *schemas.BifrostResponsesRequest) (any, error) {
			return anthropic.ToAnthropicChatRequest(ctx, c)
		}},
	{name: "anthropic_responses", provider: schemas.Anthropic, model: "claude-sonnet-4-5", responses: true, path: "tools.0.input_schema",
		build: func(ctx *schemas.BifrostContext, _ *schemas.BifrostChatRequest, r *schemas.BifrostResponsesRequest) (any, error) {
			return anthropic.ToAnthropicResponsesRequest(ctx, r)
		}},
	{name: "bedrock_claude_chat", provider: schemas.Bedrock, model: "global.anthropic.claude-sonnet-4-6", path: "toolConfig.tools.0.toolSpec.inputSchema.json",
		build: func(ctx *schemas.BifrostContext, c *schemas.BifrostChatRequest, _ *schemas.BifrostResponsesRequest) (any, error) {
			return bedrock.ToBedrockChatCompletionRequest(ctx, c)
		}},
	{name: "bedrock_nova_chat", provider: schemas.Bedrock, model: "us.amazon.nova-pro-v1:0", path: "toolConfig.tools.0.toolSpec.inputSchema.json",
		build: func(ctx *schemas.BifrostContext, c *schemas.BifrostChatRequest, _ *schemas.BifrostResponsesRequest) (any, error) {
			return bedrock.ToBedrockChatCompletionRequest(ctx, c)
		}},
	{name: "bedrock_claude_responses", provider: schemas.Bedrock, model: "global.anthropic.claude-sonnet-4-6", responses: true, path: "toolConfig.tools.0.toolSpec.inputSchema.json",
		build: func(ctx *schemas.BifrostContext, _ *schemas.BifrostChatRequest, r *schemas.BifrostResponsesRequest) (any, error) {
			return bedrock.ToBedrockResponsesRequest(ctx, r)
		}},
	{name: "gemini_chat", provider: schemas.Gemini, model: "gemini-3.8-flash", path: "tools.0.functionDeclarations.0.parametersJsonSchema",
		build: func(ctx *schemas.BifrostContext, c *schemas.BifrostChatRequest, _ *schemas.BifrostResponsesRequest) (any, error) {
			return gemini.ToGeminiChatCompletionRequest(ctx, c)
		}},
	{name: "gemini_responses", provider: schemas.Gemini, model: "gemini-3.8-flash", responses: true, path: "tools.0.functionDeclarations.0.parametersJsonSchema",
		build: func(ctx *schemas.BifrostContext, _ *schemas.BifrostChatRequest, r *schemas.BifrostResponsesRequest) (any, error) {
			return gemini.ToGeminiResponsesRequest(ctx, r)
		}},
}

// BenchmarkNormalizeChatToolSchemas_NoEmptyCombinator measures the common case, ordinary
// tools on a model with no schema row: the request is returned unchanged. The one
// allocation is the model caps resolution that predates the empty-combinator scan; the
// scan itself is allocation-free (2 ns for one tool).
func BenchmarkNormalizeChatToolSchemas_NoEmptyCombinator(b *testing.B) {
	var params schemas.ToolFunctionParameters
	if err := schemas.Unmarshal([]byte(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`), &params); err != nil {
		b.Fatal(err)
	}
	tools := make([]schemas.ChatTool, 8)
	for i := range tools {
		tools[i] = schemas.ChatTool{Type: schemas.ChatToolTypeFunction, Function: &schemas.ChatToolFunction{Name: "f", Parameters: &params}}
	}
	req := &schemas.BifrostChatRequest{Model: "gpt-5-mini", Params: &schemas.ChatParameters{Tools: tools}}
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	b.ReportAllocs()
	for b.Loop() {
		if normalizeChatToolSchemas(ctx, schemas.OpenAI, req) != req {
			b.Fatal("an ordinary request must be returned unchanged")
		}
	}
}

// TestEmptyRootCombinator_EveryWire pins BIT-2466 delta 1 as shape x wire -> the tool
// schema the wire sends, plus the tool_schema adaptation bit. Vertex Claude and Azure
// Claude share the Anthropic converter, Vertex Gemini the Gemini one, Azure GPT the
// OpenAI one.
func TestEmptyRootCombinator_EveryWire(t *testing.T) {
	const city = `"properties":{"city":{"type":"string"}}`
	for _, w := range emptyCombinatorWires {
		for _, tc := range []struct {
			shape, want string
			bit         bool
		}{
			{`{"type":"object",` + city + `,"anyOf":[]}`, `{"type":"object",` + city + `}`, true},
			{`{"anyOf":[]}`, `{"type":"object","properties":{}}`, true},
			{`{"oneOf":[],"allOf":[]}`, `{"type":"object","properties":{}}`, true},
			{`{"type":"object",` + city + `,"required":[],"enum":[]}`, `{"type":"object",` + city + `,"required":[],"enum":[]}`, false},
			{`{"type":"object",` + city + `,"required":[],"oneOf":[]}`, `{"type":"object",` + city + `,"required":[]}`, true},
		} {
			shape := tc.shape
			t.Run(w.name+"/"+shape, func(t *testing.T) {
				var params schemas.ToolFunctionParameters
				if err := schemas.Unmarshal([]byte(shape), &params); err != nil {
					t.Fatal(err)
				}
				ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
				user := "hi"
				var body any
				var err error
				if w.responses {
					r := normalizeResponsesToolSchemas(ctx, w.provider, &schemas.BifrostResponsesRequest{Provider: w.provider, Model: w.model,
						Input: []schemas.ResponsesMessage{{Role: schemas.Ptr(schemas.ResponsesInputMessageRoleUser), Content: &schemas.ResponsesMessageContent{ContentStr: &user}}},
						Params: &schemas.ResponsesParameters{Tools: []schemas.ResponsesTool{{
							Type: schemas.ResponsesToolTypeFunction, Name: new("get_weather"), ResponsesToolFunction: &schemas.ResponsesToolFunction{Parameters: &params},
						}}}})
					body, err = w.build(ctx, nil, r)
				} else {
					c := normalizeChatToolSchemas(ctx, w.provider, &schemas.BifrostChatRequest{Provider: w.provider, Model: w.model,
						Input: []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: &user}}},
						Params: &schemas.ChatParameters{Tools: []schemas.ChatTool{{
							Type: schemas.ChatToolTypeFunction, Function: &schemas.ChatToolFunction{Name: "get_weather", Parameters: &params},
						}}}})
					body, err = w.build(ctx, c, nil)
				}
				if err != nil {
					t.Fatalf("convert: %v", err)
				}
				wire, err := json.Marshal(body)
				if err != nil {
					t.Fatal(err)
				}
				if got := gjson.GetBytes(wire, w.path).Raw; got != tc.want {
					t.Errorf("tool schema on the wire = %s, want %s", got, tc.want)
				}
				if got := schemas.AdaptationsFromContext(ctx).Has(schemas.AdaptationToolSchema); got != tc.bit {
					t.Errorf("tool_schema bit = %v, want %v", got, tc.bit)
				}
			})
		}
	}
}

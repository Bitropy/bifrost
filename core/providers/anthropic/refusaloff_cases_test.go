package anthropic

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

// This file builds only with APIs that predate BIT-2954, so the same cases can produce the golden
// bodies on the base commit: refusal mode off must leave every body byte-identical to them.

const refusalOffGolden = "testdata/bit2954_refusal_off_golden.json"

// Every § Refusals feature at once, as chat parameters.
const refusalOffChatParams = `{
	"container":{"id":"c1","skills":[{"type":"anthropic","skill_id":"pdf"}]},
	"mcp_servers":[{"type":"url","url":"https://mcp.example","name":"m"}],
	"speed":"fast",
	"task_budget":{"type":"tokens","total":1000},
	"inference_geo":"us",
	"cache_control":{"type":"ephemeral","scope":"global"},
	"context_management":{"edits":[{"type":"compact_20260112"},{"type":"clear_tool_uses_20250919"}]},
	"service_tier":"auto",
	"temperature":0.5,
	"top_p":0.9,
	"tools":[
		{"type":"function","function":{"name":"f","parameters":{"type":"object"},"strict":true},"defer_loading":true,"allowed_callers":["direct"],"input_examples":[{"input":{"q":"x"}}],"eager_input_streaming":true},
		{"type":"web_fetch_20250910","name":"web_fetch"},
		{"type":"web_search_20260209","name":"web_search"}
	]
}`

// The same features as top-level raw keys, as an embedder's captured unknown keys.
var refusalOffRawKeys = map[string]string{
	"container":             `{"id":"c1","skills":[{"type":"anthropic","skill_id":"pdf"}]}`,
	"mcp_servers":           `[{"type":"url","url":"https://mcp.example","name":"m"}]`,
	"speed":                 `"fast"`,
	"inference_geo":         `"us"`,
	"diagnostics":           `{"cache_diagnosis":true}`,
	"safeguards":            `{"mode":"auto"}`,
	"fallback_credit_token": `"tok"`,
	"context_management":    `{"edits":[{"type":"compact_20260112"}]}`,
	"top_k":                 `5`,
	"x_unknown":             `{"a":1}`,
}

const refusalOffRawBody = `{"model":"claude-opus-4-7","max_tokens":64,"messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral","scope":"global"}}]}],` +
	`"container":"c1","mcp_servers":[{"type":"url","url":"u","name":"m"}],"speed":"fast","inference_geo":"us","diagnostics":{},"safeguards":{},` +
	`"fallback_credit_token":"t","service_tier":"auto","thinking":{"type":"enabled","budget_tokens":2048},"output_config":{"effort":"max","task_budget":{"type":"tokens","total":1}},` +
	`"context_management":{"edits":[{"type":"compact_20260112"}]},"tools":[{"name":"f","input_schema":{},"defer_loading":true,"allowed_callers":["direct"],"input_examples":[{"input":{}}],"eager_input_streaming":true,"strict":true},{"type":"web_search_20260209","name":"web_search"}]}`

// refusalOffBodies builds every case and returns the bodies by case name. setup prepares each
// call's context (the base commit passes a no-op).
func refusalOffBodies(t *testing.T, setup func(*schemas.BifrostContext)) map[string]string {
	t.Helper()
	out := map[string]string{}
	newCtx := func(passthrough, raw bool) *schemas.BifrostContext {
		ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
		if passthrough {
			ctx.SetValue(schemas.BifrostContextKeyPassthroughExtraParams, true)
		}
		if raw {
			ctx.SetValue(schemas.BifrostContextKeyUseRawRequestBody, true)
		}
		setup(ctx)
		return ctx
	}
	rawExtras := func() map[string]any {
		m := make(map[string]any, len(refusalOffRawKeys))
		for k, v := range refusalOffRawKeys {
			m[k] = json.RawMessage(v)
		}
		return m
	}
	betas := map[string]schemas.SecretVar{AnthropicBetaHeader: *schemas.NewSecretVar("mcp-client-2025-11-20,unknown-beta-2099")}
	for _, provider := range []schemas.ModelProvider{schemas.Anthropic, schemas.Azure, schemas.Vertex} {
		for _, model := range []string{"claude-sonnet-4-5", "claude-opus-4-7"} {
			cfg := AnthropicRequestBuildConfig{Provider: provider, Model: model, ProviderExtraHeaders: betas}
			msgs := []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("hi")}}}
			name := string(provider) + "/" + model

			typed := &schemas.ChatParameters{}
			if err := json.Unmarshal([]byte(refusalOffChatParams), typed); err != nil {
				t.Fatal(err)
			}
			typed.ExtraParams = map[string]any{"diagnostics": map[string]any{"cache_diagnosis": true}, "safeguards": json.RawMessage(`{"mode":"auto"}`), "fallback_credit_token": "tok"}
			body, bifrostErr := BuildAnthropicChatRequestBody(newCtx(false, false), &schemas.BifrostChatRequest{Provider: provider, Model: model, Input: msgs, Params: typed}, cfg)
			if bifrostErr != nil {
				t.Fatalf("%s chat typed: %s", name, bifrostErr.GetErrorString())
			}
			out[name+"/chat/typed"] = string(body)

			body, bifrostErr = BuildAnthropicChatRequestBody(newCtx(true, false), &schemas.BifrostChatRequest{Provider: provider, Model: model, Input: msgs, Params: &schemas.ChatParameters{ExtraParams: rawExtras()}}, cfg)
			if bifrostErr != nil {
				t.Fatalf("%s chat raw keys: %s", name, bifrostErr.GetErrorString())
			}
			out[name+"/chat/merged"] = string(body)

			body, bifrostErr = BuildAnthropicChatRequestBody(newCtx(false, true), &schemas.BifrostChatRequest{Provider: provider, Model: model, RawRequestBody: []byte(refusalOffRawBody)}, cfg)
			if bifrostErr != nil {
				t.Fatalf("%s chat raw body: %s", name, bifrostErr.GetErrorString())
			}
			out[name+"/chat/rawbody"] = string(body)

			rin := []schemas.ResponsesMessage{{Role: schemas.Ptr(schemas.ResponsesInputMessageRoleUser), Content: &schemas.ResponsesMessageContent{ContentStr: schemas.Ptr("hi")}}}
			rparams := &schemas.ResponsesParameters{
				Temperature: schemas.Ptr(0.5), TopP: schemas.Ptr(0.9), ServiceTier: schemas.Ptr(schemas.BifrostServiceTier("auto")),
				ContextManagement: json.RawMessage(`{"edits":[{"type":"compact_20260112"}]}`),
				ExtraParams: map[string]any{"speed": "fast", "inference_geo": "us", "top_k": 5, "fallback_credit_token": "tok",
					"container": "c1", "task_budget": map[string]any{"type": "tokens", "total": 1000}},
				Tools: []schemas.ResponsesTool{{Type: schemas.ResponsesToolTypeWebFetch}},
			}
			rcfg := cfg
			rcfg.ValidateTools = true
			body, bifrostErr = BuildAnthropicResponsesRequestBody(newCtx(false, false), &schemas.BifrostResponsesRequest{Provider: provider, Model: model, Input: rin, Params: rparams}, rcfg)
			if bifrostErr != nil {
				t.Fatalf("%s responses typed: %s", name, bifrostErr.GetErrorString())
			}
			out[name+"/responses/typed"] = string(body)

			// A raw fallback_credit_token on Responses is excluded: the base drops it at conversion
			// (it only takes a string), BIT-2954 decodes it (TestToAnthropicResponses_RawFallbackCreditToken).
			responsesRaw := rawExtras()
			delete(responsesRaw, "fallback_credit_token")
			body, bifrostErr = BuildAnthropicResponsesRequestBody(newCtx(true, false), &schemas.BifrostResponsesRequest{Provider: provider, Model: model, Input: rin, Params: &schemas.ResponsesParameters{ExtraParams: responsesRaw}}, cfg)
			if bifrostErr != nil {
				t.Fatalf("%s responses raw keys: %s", name, bifrostErr.GetErrorString())
			}
			out[name+"/responses/merged"] = string(body)
		}
	}
	return out
}

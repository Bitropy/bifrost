// Package anthropictest holds the explicit Anthropic feature rows the anthropic provider's
// carry tests and core's forwarding tests both drive, so the two read one feature list.
//
// It must not import providers/anthropic: that package's own tests import it.
package anthropictest

import (
	"encoding/json"

	"github.com/maximhq/bifrost/core/schemas"
)

// FeatureRow is one explicit feature as a request carries it, each member alone on a minimal
// request.
type FeatureRow struct {
	// Name is the feature's name, schemas.AnthropicFeature.First of its bit.
	Name string
	// Chat and Responses add the member in the shape each converter takes; Responses nil means the
	// Responses shape carries it only as a raw key (Raw).
	Chat      func(*schemas.BifrostChatRequest)
	Responses func(*schemas.BifrostResponsesRequest)
	// Raw is the member as a top-level ExtraParams key, merged under passthrough, when it has one.
	Raw   map[string]string
	Betas []string
}

// Feature is the row's bit.
func (r FeatureRow) Feature() schemas.AnthropicFeature {
	for f := schemas.AnthropicFeature(1); f != 0; f <<= 1 {
		if f.First() == r.Name {
			return f
		}
	}
	panic("anthropictest: unknown feature " + r.Name)
}

// MCPClientBeta is the client anthropic-beta value the anthropic_beta row sends.
const MCPClientBeta = "mcp-client-2025-11-20"

// FeatureRows are the explicit features in bit order. Every value is a probe no baseline request
// holds, so a test can look for it in an upstream body.
func FeatureRows() []FeatureRow {
	tool := func(flag string) func(*schemas.BifrostChatRequest) {
		return ChatParams(`{"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}},` + flag + `}]}`)
	}
	respFn := func(flag string) func(*schemas.BifrostResponsesRequest) {
		return RespTool(`{"type":"function","name":"f","parameters":{"type":"object"},` + flag + `}`)
	}
	return []FeatureRow{
		{Name: "container", Chat: ChatParams(`{"container":"cprobe1"}`), Responses: RespExtra("container", json.RawMessage(`"cprobe1"`)),
			Raw: map[string]string{"container": `"cprobe1"`}},
		{Name: "container_skills", Chat: ChatParams(`{"container":{"id":"c1","skills":[{"type":"anthropic","skill_id":"skillprobe"}]}}`),
			Responses: RespExtra("container", json.RawMessage(`{"id":"c1","skills":[{"type":"anthropic","skill_id":"skillprobe"}]}`)),
			Raw:       map[string]string{"container": `{"id":"c1","skills":[{"type":"anthropic","skill_id":"skillprobe"}]}`}},
		{Name: "mcp_servers", Chat: ChatParams(`{"mcp_servers":[{"type":"url","url":"https://mcpprobe.example","name":"m"}]}`),
			Raw: map[string]string{"mcp_servers": `[{"type":"url","url":"https://mcpprobe.example","name":"m"}]`}},
		{Name: "speed", Chat: ChatParams(`{"speed":"fast"}`), Responses: RespExtra("speed", "fast"),
			Raw: map[string]string{"speed": `"fast"`}},
		{Name: "task_budget", Chat: ChatParams(`{"task_budget":{"type":"tokens","total":4242}}`),
			Responses: RespExtra("task_budget", json.RawMessage(`{"type":"tokens","total":4242}`))},
		{Name: "inference_geo", Chat: ChatParams(`{"inference_geo":"us"}`), Responses: RespExtra("inference_geo", "us"),
			Raw: map[string]string{"inference_geo": `"us"`}},
		{Name: "diagnostics", Chat: ChatExtra("diagnostics", map[string]any{"cache_diagnosis": true}),
			Responses: RespExtra("diagnostics", json.RawMessage(`{"cache_diagnosis":true}`)),
			Raw:       map[string]string{"diagnostics": `{"cache_diagnosis":true}`}},
		{Name: "safeguards", Chat: ChatExtra("safeguards", json.RawMessage(`{"mode":"auto"}`)),
			Responses: RespExtra("safeguards", json.RawMessage(`{"mode":"auto"}`)),
			Raw:       map[string]string{"safeguards": `{"mode":"auto"}`}},
		{Name: "cache_control_scope", Chat: ChatParams(`{"cache_control":{"type":"ephemeral","scope":"global"}}`),
			Responses: respFn(`"cache_control":{"type":"ephemeral","scope":"global"}`)},
		{Name: "fallback_credit_token", Chat: ChatExtra("fallback_credit_token", "tokprobe"), Responses: RespExtra("fallback_credit_token", "tokprobe"),
			Raw: map[string]string{"fallback_credit_token": `"tokprobe"`}},
		{Name: "context_management", Chat: ChatParams(`{"context_management":{"edits":[{"type":"compact_20260112"}]}}`),
			Responses: RespExtra("context_management", json.RawMessage(`{"edits":[{"type":"compact_20260112"}]}`)),
			Raw:       map[string]string{"context_management": `{"edits":[{"type":"compact_20260112"}]}`}},
		{Name: "defer_loading", Chat: tool(`"defer_loading":true`), Responses: respFn(`"defer_loading":true`)},
		{Name: "allowed_callers", Chat: tool(`"allowed_callers":["direct"]`), Responses: respFn(`"allowed_callers":["direct"]`)},
		{Name: "input_examples", Chat: tool(`"input_examples":[{"input":{"q":"x"}}]`), Responses: respFn(`"input_examples":[{"input":{"q":"x"}}]`)},
		{Name: "eager_input_streaming", Chat: tool(`"eager_input_streaming":true`), Responses: respFn(`"eager_input_streaming":true`)},
		{Name: "server_tool", Chat: ChatParams(`{"tools":[{"type":"web_fetch_20250910","name":"web_fetch"}]}`), Responses: RespTool(`{"type":"web_fetch"}`)},
		{Name: "tool_version", Chat: ChatParams(`{"tools":[{"type":"web_search_20260209","name":"web_search"}]}`),
			Responses: RespTool(`{"type":"web_search","web_search_version":"web_search_20260209"}`)},
		{Name: "anthropic_beta", Betas: []string{MCPClientBeta}},
		{Name: "fallback_block", Chat: ChatReplay(`{"type":"fallback"}`), Responses: RespReplay(`{"type":"fallback","from_model":"fromprobe","to_model":"b"}`)},
		{Name: "compaction_block", Chat: ChatReplay(`{"type":"compaction","content":"summaryprobe"}`), Responses: RespReplay(`{"type":"compaction","summary":"summaryprobe"}`)},
		{Name: "tool_change_block", Chat: ChatReplay(`{"type":"tool_addition","tool":{"type":"tool_reference","name":"f"}}`)},
	}
}

// ChatParams unmarshals js over the request's parameters.
func ChatParams(js string) func(*schemas.BifrostChatRequest) {
	return func(r *schemas.BifrostChatRequest) {
		if err := json.Unmarshal([]byte(js), r.Params); err != nil {
			panic(err)
		}
	}
}

// ChatExtra sets the request's ExtraParams to the one key.
func ChatExtra(key string, v any) func(*schemas.BifrostChatRequest) {
	return func(r *schemas.BifrostChatRequest) {
		r.Params.ExtraParams = map[string]any{key: v}
	}
}

// RespExtra sets the request's ExtraParams to the one key.
func RespExtra(key string, v any) func(*schemas.BifrostResponsesRequest) {
	return func(r *schemas.BifrostResponsesRequest) {
		r.Params.ExtraParams = map[string]any{key: v}
	}
}

// RespTool appends the tool js decodes to.
func RespTool(js string) func(*schemas.BifrostResponsesRequest) {
	return func(r *schemas.BifrostResponsesRequest) {
		var tool schemas.ResponsesTool
		if err := json.Unmarshal([]byte(js), &tool); err != nil {
			panic(err)
		}
		r.Params.Tools = append(r.Params.Tools, tool)
	}
}

// ChatReplay appends an assistant turn replaying block as an Anthropic provider block, then a user
// turn.
func ChatReplay(block string) func(*schemas.BifrostChatRequest) {
	return func(r *schemas.BifrostChatRequest) {
		r.Input = append(r.Input, schemas.ChatMessage{
			Role:    schemas.ChatMessageRoleAssistant,
			Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("answer")},
			ChatAssistantMessage: &schemas.ChatAssistantMessage{ProviderBlocks: []schemas.ChatProviderBlock{
				{Index: 0, Provider: schemas.ProviderBlockAnthropic, Block: []byte(block)},
			}},
		}, schemas.ChatMessage{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("next")}})
	}
}

// RespReplay appends an assistant item whose content leads with block, then a user item.
func RespReplay(block string) func(*schemas.BifrostResponsesRequest) {
	return func(r *schemas.BifrostResponsesRequest) {
		var items []schemas.ResponsesMessage
		if err := json.Unmarshal([]byte(`[{"type":"message","role":"assistant","content":[`+block+`,{"type":"output_text","text":"answer"}]},`+
			`{"type":"message","role":"user","content":"next"}]`), &items); err != nil {
			panic(err)
		}
		r.Input = append(r.Input, items...)
	}
}

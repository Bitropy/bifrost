package anthropic

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

// CacheControlIngressBody is an Anthropic /v1/messages body marking a tool,
// a system block, a user text block (1h ttl) and a tool_result: four
// markers, the Anthropic limit.
const cacheControlIngressBody = `{"model":"claude-sonnet-4-5","max_tokens":256,` +
	`"tools":[{"name":"read","input_schema":{"type":"object"},"cache_control":{"type":"ephemeral"}}],` +
	`"system":[{"type":"text","text":"sys one"},{"type":"text","text":"sys two","cache_control":{"type":"ephemeral"}}],` +
	`"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b","cache_control":{"type":"ephemeral","ttl":"1h"}}]},` +
	`{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"read","input":{}}]},` +
	`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"x","cache_control":{"type":"ephemeral"}}]}]}`

// AnthropicIngressToChat is the Anthropic ingress's translated path: the
// Anthropic body to a Responses request, then the Responses→Chat fallback.
func anthropicIngressToChat(t *testing.T, body string) *schemas.BifrostChatRequest {
	t.Helper()
	var areq AnthropicMessageRequest
	require.NoError(t, sonic.Unmarshal([]byte(body), &areq))
	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	t.Cleanup(ctx.Cancel)
	return areq.ToBifrostResponsesRequest(ctx).ToChatRequest()
}

// CacheControlMarkers keys every cache_control of an Anthropic body by the
// block it sits on.
func cacheControlMarkers(t *testing.T, body []byte) map[string]string {
	t.Helper()
	out := map[string]string{}
	note := func(v gjson.Result) bool {
		if cc := v.Get("cache_control"); cc.Exists() {
			key := v.Get("type").String() + ":" + v.Get("text").String() + v.Get("id").String() + v.Get("tool_use_id").String()
			if v.Get("input_schema").Exists() {
				key = "tool:" + v.Get("name").String()
			}
			out[key] = cc.Raw
		}
		return true
	}
	for _, path := range []string{"tools", "system", "messages.#.content|@flatten"} {
		gjson.GetBytes(body, path).ForEach(func(_, v gjson.Result) bool { return note(v) })
	}
	if cc := gjson.GetBytes(body, "cache_control"); cc.Exists() {
		out["request"] = cc.Raw
	}
	return out
}

// BIT-2764: every marker of an Anthropic ingress body reaches each Anthropic
// wire behind a chat route — native, Azure and Vertex Claude — through the
// Responses→Chat fallback, ttl included; a tool_use's and the request-level
// marker too.
func TestAnthropicIngressCacheControl_ReachesEveryAnthropicWire(t *testing.T) {
	const eph = `{"type":"ephemeral"}`
	toolUseBody := strings.NewReplacer(
		`"max_tokens":256,`, `"max_tokens":256,"cache_control":{"type":"ephemeral"},`,
		`"input_schema":{"type":"object"},"cache_control":{"type":"ephemeral"}`, `"input_schema":{"type":"object"}`,
		`"text":"sys two","cache_control":{"type":"ephemeral"}`, `"text":"sys two"`,
		`"input":{}}`, `"input":{},"cache_control":{"type":"ephemeral"}}`,
	).Replace(cacheControlIngressBody)
	for name, tc := range map[string]struct {
		body string
		want map[string]string
	}{
		"blocks and tool": {cacheControlIngressBody, map[string]string{
			"tool:read": eph, "text:sys two": eph, "text:b": `{"type":"ephemeral","ttl":"1h"}`, "tool_result:toolu_1": eph,
		}},
		"tool_use and request": {toolUseBody, map[string]string{
			"text:b": `{"type":"ephemeral","ttl":"1h"}`, "tool_use:toolu_1": eph, "tool_result:toolu_1": eph, "request": eph,
		}},
	} {
		for _, w := range []struct {
			provider schemas.ModelProvider
			model    string
		}{
			{schemas.Anthropic, "claude-sonnet-4-5"},
			{schemas.Azure, "claude-sonnet-4-5"},
			{schemas.Vertex, "claude-sonnet-4-5@20250929"},
		} {
			t.Run(name+"/"+string(w.provider), func(t *testing.T) {
				req := anthropicIngressToChat(t, tc.body)
				req.Provider, req.Model = w.provider, w.model
				ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
				defer ctx.Cancel()
				body, berr := BuildAnthropicChatRequestBody(ctx, req, AnthropicRequestBuildConfig{Provider: w.provider, Model: w.model})
				require.Nil(t, berr)
				assert.Equal(t, tc.want, cacheControlMarkers(t, body), "%s", body)
			})
		}
	}
}

// A chat tool message's marker rides its first content block. Through the
// chat→Responses→Anthropic path (a chat request served on the Responses API,
// e.g. an Azure Claude deployment that requires it) it must land on the
// tool_result, never inside tool_result.content, which Anthropic rejects.
func TestChatToolResultCacheControl_HoistedOnResponsesPath(t *testing.T) {
	eph := &schemas.CacheControl{Type: schemas.CacheControlTypeEphemeral}
	chat := &schemas.BifrostChatRequest{
		Provider: schemas.Azure, Model: "claude-sonnet-4-5",
		Input: []schemas.ChatMessage{
			{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("q")}},
			{Role: schemas.ChatMessageRoleAssistant, ChatAssistantMessage: &schemas.ChatAssistantMessage{ToolCalls: []schemas.ChatAssistantMessageToolCall{
				{ID: schemas.Ptr("toolu_1"), Type: schemas.Ptr("function"), Function: schemas.ChatAssistantMessageToolCallFunction{Name: schemas.Ptr("read"), Arguments: "{}"}},
			}}},
			{Role: schemas.ChatMessageRoleTool, ChatToolMessage: &schemas.ChatToolMessage{ToolCallID: schemas.Ptr("toolu_1")},
				Content: &schemas.ChatMessageContent{ContentBlocks: []schemas.ChatContentBlock{{Type: schemas.ChatContentBlockTypeText, Text: schemas.Ptr("x"), CacheControl: eph}}}},
		},
		Params: &schemas.ChatParameters{MaxCompletionTokens: schemas.Ptr(256)},
	}
	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	defer ctx.Cancel()
	out, berr := BuildAnthropicResponsesRequestBody(ctx, chat.ToResponsesRequest(), AnthropicRequestBuildConfig{Provider: schemas.Azure, Model: "claude-sonnet-4-5"})
	require.Nil(t, berr)
	result := gjson.GetBytes(out, `messages.#.content|@flatten|#(type=="tool_result")`)
	require.True(t, result.Exists(), "%s", out)
	assert.JSONEq(t, `{"type":"ephemeral"}`, result.Get("cache_control").Raw, "%s", out)
	assert.NotContains(t, result.Get("content").Raw, "cache_control", "%s", out)
}

// An Azure Claude deployment served through the Responses API (it requires
// it): the Anthropic ingress body's markers reach its Anthropic body through
// chat→Responses→Anthropic, the tool_result's on the tool_result (BIT-2764).
func TestAnthropicIngressCacheControl_ResponsesPathToClaude(t *testing.T) {
	const eph = `{"type":"ephemeral"}`
	chat := anthropicIngressToChat(t, cacheControlIngressBody)
	chat.Provider, chat.Model = schemas.Azure, "claude-sonnet-4-5"
	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	defer ctx.Cancel()
	out, berr := BuildAnthropicResponsesRequestBody(ctx, chat.ToResponsesRequestWithCacheControl(), AnthropicRequestBuildConfig{Provider: schemas.Azure, Model: "claude-sonnet-4-5"})
	require.Nil(t, berr)
	assert.Equal(t, map[string]string{
		"tool:read": eph, "text:sys two": eph, "text:b": `{"type":"ephemeral","ttl":"1h"}`, "tool_result:toolu_1": eph,
	}, cacheControlMarkers(t, out), "%s", out)
	assert.NotContains(t, gjson.GetBytes(out, `messages.#.content|@flatten|#(type=="tool_result").content`).Raw, "cache_control")
}

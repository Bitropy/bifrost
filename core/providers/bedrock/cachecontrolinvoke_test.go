package bedrock

import (
	"context"
	"testing"
	"time"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/core/providers/anthropic"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

// BIT-2764: an Anthropic ingress body's markers reach Bedrock Claude's
// InvokeModel body through the Responses→Chat fallback.
func TestAnthropicIngressCacheControl_BedrockInvoke(t *testing.T) {
	body := `{"model":"m","max_tokens":256,` +
		`"tools":[{"name":"read","input_schema":{"type":"object"},"cache_control":{"type":"ephemeral"}}],` +
		`"system":[{"type":"text","text":"sys","cache_control":{"type":"ephemeral"}}],` +
		`"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b","cache_control":{"type":"ephemeral","ttl":"1h"}}]},` +
		`{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"read","input":{}}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"x","cache_control":{"type":"ephemeral"}}]}]}`
	var areq anthropic.AnthropicMessageRequest
	require.NoError(t, sonic.Unmarshal([]byte(body), &areq))
	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	defer ctx.Cancel()
	req := areq.ToBifrostResponsesRequest(ctx).ToChatRequest()
	const model = "us.anthropic.claude-sonnet-4-6"
	req.Provider, req.Model = schemas.Bedrock, model
	out, berr := anthropic.BuildAnthropicChatRequestBody(ctx, req, (&BedrockProvider{}).invokeBuildConfig(model, false, true, nil))
	require.Nil(t, berr)
	const eph = `{"type":"ephemeral"}`
	assert.JSONEq(t, eph, gjson.GetBytes(out, "tools.0.cache_control").Raw, "%s", out)
	assert.JSONEq(t, eph, gjson.GetBytes(out, "system.0.cache_control").Raw, "%s", out)
	assert.JSONEq(t, `{"type":"ephemeral","ttl":"1h"}`, gjson.GetBytes(out, `messages.#.content|@flatten|#(text=="b").cache_control`).Raw, "%s", out)
	assert.JSONEq(t, eph, gjson.GetBytes(out, `messages.#.content|@flatten|#(type=="tool_result").cache_control`).Raw, "%s", out)
}

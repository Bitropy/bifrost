package openai

import (
	"strings"
	"testing"

	"github.com/bytedance/sonic"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
)

// decodeChatMessages decodes OpenAI-shaped messages the way the chat ingress
// does, so a message-level cache_control is attached by AttachMessageCacheControl.
func decodeChatMessages(t *testing.T, js string) []schemas.ChatMessage {
	t.Helper()
	var msgs []schemas.ChatMessage
	require.NoError(t, sonic.Unmarshal([]byte(js), &msgs))
	return msgs
}

const ccEph = `"cache_control":{"type":"ephemeral"}`

func ccMarked(n int) string {
	parts := make([]string, n)
	for i := range parts {
		parts[i] = `{"role":"user","content":"m` + string(rune('a'+i)) + `",` + ccEph + `}`
	}
	return "[" + strings.Join(parts, ",") + "]"
}

// TestCacheControlAdaptation_Chat: the chat wire reports AdaptationUnmapped exactly
// when a client cache_control does not reach the wire as sent or as a breakpoint.
func TestCacheControlAdaptation_Chat(t *testing.T) {
	marked := `[{"role":"system","content":"x",` + ccEph + `},{"role":"user","content":"hi"}]`
	rows := []struct {
		name, msgs         string
		provider           schemas.ModelProvider
		base               schemas.ModelProvider // custom key's base; "" for a native key
		model              string
		params             *schemas.ChatParameters
		wantHas, wantLacks []string
		wantCounts         map[string]int
		wantUnmapped       bool
	}{
		{name: "gpt-4o strips", msgs: marked, provider: schemas.OpenAI, model: "gpt-4o", wantLacks: []string{"cache_control"}, wantUnmapped: true},
		{name: "azure gpt-4o strips", msgs: marked, provider: schemas.Azure, model: "gpt-4o", wantLacks: []string{"cache_control"}, wantUnmapped: true},
		{name: "mistral key strips", msgs: marked, provider: schemas.Mistral, model: "mistral-large", wantLacks: []string{"cache_control"}, wantUnmapped: true},
		{name: "breakpoint model converts", msgs: marked, provider: schemas.OpenAI, model: "gpt-6.2-sol", wantHas: []string{"prompt_cache_breakpoint"}, wantLacks: []string{"cache_control"}},
		{name: "breakpoint typeless marker strips", msgs: `[{"role":"user","content":"x","cache_control":{}}]`, provider: schemas.OpenAI, model: "gpt-6.2-sol", wantLacks: []string{"cache_control", "prompt_cache_breakpoint"}, wantUnmapped: true},
		{name: "breakpoint last part image strips", msgs: `[{"role":"user","content":[{"type":"text","text":"a"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}],` + ccEph + `}]`, provider: schemas.OpenAI, model: "gpt-6.2-sol", wantLacks: []string{"cache_control", "prompt_cache_breakpoint"}, wantUnmapped: true},
		{name: "breakpoint tool_calls-only strips", msgs: `[{"role":"assistant","tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":"{}"}}],` + ccEph + `}]`, provider: schemas.OpenAI, model: "gpt-6.2-sol", wantLacks: []string{"cache_control"}, wantUnmapped: true},
		{name: "breakpoint over implicit ceiling", msgs: ccMarked(4), provider: schemas.OpenAI, model: "gpt-6.2-sol", wantCounts: map[string]int{"prompt_cache_breakpoint": 3}, wantLacks: []string{"cache_control"}, wantUnmapped: true},
		{name: "breakpoint explicit ceiling", msgs: ccMarked(4), provider: schemas.OpenAI, model: "gpt-6.2-sol", params: &schemas.ChatParameters{PromptCacheOptions: &schemas.PromptCacheOptions{Mode: schemas.Ptr(PromptCacheBreakpointModeExplicit)}}, wantCounts: map[string]int{"prompt_cache_breakpoint": 4}},
		{name: "openrouter claude carries", msgs: marked, provider: schemas.OpenRouter, model: "anthropic/claude-haiku-4.5", wantHas: []string{`"text":"x"`, ccEph}},
		{name: "openrouter claude clamps a fifth", msgs: ccMarked(5), provider: schemas.OpenRouter, model: "anthropic/claude-haiku-4.5", wantCounts: map[string]int{"cache_control": 4}, wantUnmapped: true},
		{name: "openrouter base under another key strips", msgs: marked, provider: "my-router", base: schemas.OpenRouter, model: "anthropic/claude-haiku-4.5", wantLacks: []string{"cache_control"}, wantUnmapped: true},
		{name: "null message marker", msgs: `[{"role":"user","content":"x","cache_control":null}]`, provider: schemas.OpenAI, model: "gpt-4o"},
		{name: "null part marker leaves the message marker", msgs: `[{"role":"user","content":[{"type":"text","text":"x","cache_control":null}],` + ccEph + `}]`, provider: schemas.OpenRouter, model: "anthropic/claude-haiku-4.5", wantHas: []string{`"text":"x",` + ccEph}},
		{name: "no marker", msgs: `[{"role":"user","content":"x"}]`, provider: schemas.OpenAI, model: "gpt-4o"},
		{name: "tool marker strips", msgs: `[{"role":"user","content":"x"}]`, provider: schemas.OpenAI, model: "gpt-4o",
			params:    &schemas.ChatParameters{Tools: []schemas.ChatTool{{Type: schemas.ChatToolTypeFunction, Function: &schemas.ChatToolFunction{Name: "f"}, CacheControl: &schemas.CacheControl{Type: schemas.CacheControlTypeEphemeral}}}},
			wantLacks: []string{"cache_control"}, wantUnmapped: true},
	}
	for _, row := range rows {
		t.Run(row.name, func(t *testing.T) {
			ctx := samplingCtx(t)
			if row.base != "" {
				ctx.SetValue(schemas.BifrostContextKeyBaseProviderType, row.base)
			}
			req := ToOpenAIChatRequest(ctx, &schemas.BifrostChatRequest{Provider: row.provider, Model: row.model, Input: decodeChatMessages(t, row.msgs), Params: row.params})
			require.NotNil(t, req)
			out, err := providerUtils.MarshalSorted(req)
			require.NoError(t, err)
			wire := string(out)
			for _, s := range row.wantHas {
				assert.Contains(t, wire, s)
			}
			for _, s := range row.wantLacks {
				assert.NotContains(t, wire, s)
			}
			for s, n := range row.wantCounts {
				assert.Equal(t, n, strings.Count(wire, s), s)
			}
			assert.Equal(t, row.wantUnmapped, schemas.AdaptationsFromContext(ctx).Has(schemas.AdaptationUnmapped), wire)
		})
	}
}

// TestCacheControlAdaptation_Responses: the chat→Responses translation and the
// Responses serializer report a marker they lose, never one turned into a breakpoint.
func TestCacheControlAdaptation_Responses(t *testing.T) {
	marked := `[{"role":"system","content":"x",` + ccEph + `},{"role":"user","content":"hi"}]`
	toolResult := `[{"role":"user","content":"q"},{"role":"assistant","tool_calls":[{"id":"t1","type":"function","function":{"name":"f","arguments":"{}"}}]},{"role":"tool","tool_call_id":"t1","content":"ok",` + ccEph + `}]`
	rows := []struct {
		name, msgs, model  string
		wantHas, wantLacks []string
		wantUnmapped       bool
	}{
		{name: "gpt-4o drops in translation", msgs: marked, model: "gpt-4o", wantLacks: []string{"cache_control", "prompt_cache_breakpoint"}, wantUnmapped: true},
		{name: "breakpoint model drops in translation", msgs: marked, model: "gpt-6.2-sol", wantLacks: []string{"cache_control", "prompt_cache_breakpoint"}, wantUnmapped: true},
		{name: "claude on an OpenAI base strips", msgs: marked, model: "claude-haiku-4.5", wantLacks: []string{"cache_control", "prompt_cache_breakpoint"}, wantUnmapped: true},
		{name: "tool result becomes a breakpoint", msgs: toolResult, model: "gpt-6.2-sol", wantHas: []string{"prompt_cache_breakpoint"}, wantLacks: []string{"cache_control"}},
		{name: "null marker", msgs: `[{"role":"user","content":"x","cache_control":null}]`, model: "gpt-4o"},
	}
	for _, row := range rows {
		t.Run(row.name, func(t *testing.T) {
			chat := &schemas.BifrostChatRequest{Provider: "my-openai", Model: row.model, Input: decodeChatMessages(t, row.msgs)}
			wire, a := openAIResponsesBody(t, chat)
			for _, s := range row.wantHas {
				assert.Contains(t, wire, s)
			}
			for _, s := range row.wantLacks {
				assert.NotContains(t, wire, s)
			}
			assert.Equal(t, row.wantUnmapped, a.Has(schemas.AdaptationUnmapped), wire)
		})
	}
}

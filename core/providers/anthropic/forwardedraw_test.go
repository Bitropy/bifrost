package anthropic

import (
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

// The conversion tables are read on the request path: the lookup allocates nothing, on every
// dispatch branch. Their contents are pinned against the converters by
// TestForwardedRaw_AgreesWithConverters and TestConsumedExtraKeys_AgreesWithConverters (core).
func TestConversionLookup_AllocBudget(t *testing.T) {
	targets := []struct {
		provider schemas.ModelProvider
		model    string
	}{
		{schemas.Vertex, "gemini-2.5-flash"}, {schemas.Vertex, "meta/llama-3.3-70b-instruct-maas"}, {schemas.Vertex, "claude-sonnet-4-5"},
		{schemas.Azure, "gpt-4o"}, {schemas.Perplexity, "sonar"}, {schemas.GithubCopilot, "gpt-4o"}, {schemas.Gemini, "gemini-2.5-flash"},
		{schemas.DeepSeek, "deepseek-chat"},
	}
	for _, tg := range targets {
		allocs := testing.AllocsPerRun(50, func() {
			for _, kind := range []schemas.RequestType{schemas.ChatCompletionRequest, schemas.ResponsesRequest} {
				_ = ForwardedRaw(tg.provider, tg.model, kind, true)
				_ = ConsumedExtraKeys(tg.provider, tg.model, kind)
				_ = ConsumesAllExtras(tg.provider, tg.model, kind)
			}
		})
		if allocs != 0 {
			t.Errorf("%s/%s: allocs = %v", tg.provider, tg.model, allocs)
		}
	}
	if ForwardedRaw("unknown", "m", schemas.ChatCompletionRequest, true) != 0 || ConsumedExtraKeys("unknown", "m", schemas.ChatCompletionRequest) != nil {
		t.Fatal("a provider with no row forwards and reads nothing")
	}
	if ConsumedExtraKeys(schemas.Gemini, "gemini-2.5-flash", schemas.EmbeddingRequest) != nil ||
		ConsumesAllExtras(schemas.DeepSeek, "deepseek-chat", schemas.EmbeddingRequest) {
		t.Fatal("only chat and Responses have rows")
	}
	if !ConsumesAllExtras(schemas.DeepSeek, "deepseek-chat", schemas.ChatCompletionRequest) ||
		ConsumedExtraKeys(schemas.DeepSeek, "deepseek-chat", schemas.ChatCompletionRequest) != nil ||
		ConsumesAllExtras(schemas.Gemini, "gemini-2.5-flash", schemas.ChatCompletionRequest) {
		t.Fatal("a merging conversion consumes every key and names none; a listing one does not consume all")
	}
}

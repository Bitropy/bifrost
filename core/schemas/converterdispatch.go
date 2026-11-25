package schemas

import "strings"

// IsVertexGeminiModel reports whether Vertex sends model's chat and Responses requests through
// the Gemini converter (publishers/google): a Gemini or Gemma model, or an all-digit endpoint ID.
// Vertex's dispatch and the request-conversion tables in providers/anthropic both ask it.
func IsVertexGeminiModel(ctx *BifrostContext, model string) bool {
	return IsGeminiModelFamily(ctx, model) || IsAllDigitsASCII(model) || IsGemmaModelFamily(ctx, model)
}

// PerplexityServesResponses reports whether Perplexity serves model (canonical) on /v1/responses;
// the sonar models are served on /chat/completions only.
func PerplexityServesResponses(model string) bool {
	bare := strings.ToLower(strings.TrimPrefix(model, "perplexity/"))
	return bare != "sonar" && !strings.HasPrefix(bare, "sonar-")
}

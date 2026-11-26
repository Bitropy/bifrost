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

// AzureServesResponses reports whether Azure (provider, which may be a custom provider over it)
// serves model's (canonical) Responses requests on /v1/responses. A datasheet row listing
// supported_endpoints without it routes them through chat completions; a silent or missing row
// keeps the Responses route.
func AzureServesResponses(provider ModelProvider, model string) bool {
	return ResolveModelCaps(provider, model).SupportsResponsesEndpoint(true)
}

// GithubCopilotServesResponses reports whether GitHub Copilot (provider, which may be a custom
// provider over it) serves model's (canonical) Responses requests on /responses; without a
// datasheet answer only OpenAI-family models are served there, the rest through chat completions.
func GithubCopilotServesResponses(ctx *BifrostContext, provider ModelProvider, model string) bool {
	return ResolveModelCaps(provider, model).SupportsResponsesEndpoint(IsOpenAIModelFamily(ctx, model))
}

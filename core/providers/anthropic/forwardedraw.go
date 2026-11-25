package anthropic

import "github.com/maximhq/bifrost/core/schemas"

// AllExtraKeys is ConsumedExtraKeys' one entry for a conversion that sends every ExtraParams key
// upstream whether or not passthrough is on: no key is dropped.
const AllExtraKeys = "*"

// conversion is what one request conversion does with explicit Anthropic members and ExtraParams:
// the features it sends upstream raw without and with BifrostContextKeyPassthroughExtraParams,
// and the top-level ExtraParams keys it reads with passthrough off. The values are read off the
// converters by TestForwardedRaw_AgreesWithConverters and TestConsumedExtraKeys_AgreesWithConverters
// (core package), which drive each one into a recording upstream.
type conversion struct {
	forwarded, forwardedPassthrough Feature
	consumed                        []string
}

// Feature groups the conversions forward together.
const (
	// typedChatMembers are the ChatParameters members an OpenAI-shaped chat body serializes.
	typedChatMembers = FeatureContainer | FeatureContainerSkills | FeatureMCPServers | FeatureSpeed |
		FeatureTaskBudget | FeatureInferenceGeo | FeatureContextManagement
	// extraOnlyMembers are the members a chat request carries only as ExtraParams keys.
	extraOnlyMembers = FeatureDiagnostics | FeatureSafeguards | FeatureFallbackCreditToken
	// extraMembers are every member a Responses request carries as an ExtraParams key.
	extraMembers = typedChatMembers | extraOnlyMembers
	// replayBlocks are the replayed blocks the Responses-to-chat conversion keeps as content parts.
	replayBlocks = FeatureFallbackBlock | FeatureCompactionBlock
	// toolMembers are the per-tool flags and the server tools a raw tool array carries.
	toolMembers = FeatureDeferLoading | FeatureAllowedCallers | FeatureInputExamples | FeatureEagerInputStreaming |
		FeatureServerTool | FeatureToolVersion
)

var (
	geminiChatKeys = []string{"cached_content", "labels", "response_json_schema", "response_mime_type", "safety_settings", "top_k"}
	geminiRespKeys = []string{"cached_content", "candidate_count", "media_resolution", "response_logprobs", "safety_settings", "seed", "stop_sequences", "top_k"}
	bedrockKeys    = []string{"additionalModelRequestFieldPaths", "additionalModelResponseFieldPaths", "guardrailConfig",
		"performanceConfig", "promptVariables", "requestMetadata"}
	anthropicChatKeys = []string{"context_management", "diagnostics", "fallback_credit_token", "fallbacks", "inference_geo",
		"safeguards", "speed", "thinking", "top_k"}
	anthropicRespKeys = []string{"container", "context_management", "diagnostics", "fallback_credit_token", "fallbacks",
		"inference_geo", "safeguards", "speed", "task_budget", "top_k"}
	// The hosted Claude surfaces strip the native fallbacks member silently.
	hostedClaudeChatKeys = []string{"context_management", "diagnostics", "fallback_credit_token", "inference_geo",
		"safeguards", "speed", "thinking", "top_k"}
	hostedClaudeRespKeys = []string{"container", "context_management", "diagnostics", "fallback_credit_token",
		"inference_geo", "safeguards", "speed", "task_budget", "top_k"}
	allExtraKeys = []string{AllExtraKeys}
	topKOnly     = []string{"top_k"}
)

var (
	// Claude conversions carry the members they send (Uncarried's question); none is raw.
	anthropicChat    = conversion{consumed: anthropicChatKeys}
	anthropicResp    = conversion{consumed: anthropicRespKeys}
	hostedClaudeChat = conversion{consumed: hostedClaudeChatKeys}
	hostedClaudeResp = conversion{consumed: hostedClaudeRespKeys}

	// openai.ToOpenAIChatRequest: the typed members always, ExtraParams under passthrough.
	openAIChat = conversion{forwarded: typedChatMembers | FeatureCacheControlScope, forwardedPassthrough: typedChatMembers | FeatureCacheControlScope | extraOnlyMembers}
	// An OpenAI-shaped chat conversion that merges every ExtraParams key.
	mergingChat = conversion{forwarded: typedChatMembers | FeatureCacheControlScope | extraOnlyMembers,
		forwardedPassthrough: typedChatMembers | FeatureCacheControlScope | extraOnlyMembers, consumed: allExtraKeys}
	// openai.ToOpenAIResponsesRequest: tools as sent, ExtraParams under passthrough.
	openAIResponses  = conversion{forwarded: FeatureAllowedCallers, forwardedPassthrough: extraMembers | FeatureAllowedCallers}
	mergingResponses = conversion{forwarded: extraMembers | FeatureAllowedCallers, forwardedPassthrough: extraMembers | FeatureAllowedCallers,
		consumed: allExtraKeys}
	// A Responses request served through the provider's OpenAI chat conversion (ToChatRequest).
	responsesViaChat        = conversion{forwarded: replayBlocks, forwardedPassthrough: extraMembers | replayBlocks}
	responsesViaMergingChat = conversion{forwarded: extraMembers | replayBlocks, forwardedPassthrough: extraMembers | replayBlocks,
		consumed: allExtraKeys}

	geminiChat  = conversion{forwardedPassthrough: extraOnlyMembers, consumed: geminiChatKeys}
	geminiResp  = conversion{forwardedPassthrough: extraMembers, consumed: geminiRespKeys}
	bedrockChat = conversion{forwardedPassthrough: extraOnlyMembers, consumed: bedrockKeys}
	bedrockResp = conversion{forwardedPassthrough: extraMembers, consumed: bedrockKeys}
	cohereChat  = conversion{forwarded: FeatureServerTool | FeatureToolVersion, forwardedPassthrough: extraOnlyMembers | FeatureServerTool | FeatureToolVersion,
		consumed: []string{"log_probs", "safety_mode", "strict_tool_choice", "thinking"}}
	cohereResp     = conversion{forwardedPassthrough: extraMembers, consumed: []string{"thinking", "top_k"}}
	replicateChat  = conversion{forwarded: extraOnlyMembers, forwardedPassthrough: extraOnlyMembers, consumed: allExtraKeys}
	replicateResp  = conversion{forwarded: extraMembers, forwardedPassthrough: extraMembers, consumed: allExtraKeys}
	perplexityChat = conversion{forwarded: toolMembers | replayBlocks | FeatureToolChangeBlock,
		forwardedPassthrough: extraOnlyMembers | toolMembers | replayBlocks | FeatureToolChangeBlock, consumed: topKOnly}
	perplexityRespViaChat = conversion{forwarded: replayBlocks, forwardedPassthrough: extraMembers | replayBlocks, consumed: topKOnly}
	huggingFaceChat       = conversion{forwarded: toolMembers, forwardedPassthrough: extraOnlyMembers | toolMembers}
	basetenChat           = conversion{forwarded: extraOnlyMembers, forwardedPassthrough: extraOnlyMembers, consumed: allExtraKeys}
)

// conversionFor is the conversion core runs for a chat or Responses request to provider and
// model (canonical), chosen with the providers' own dispatch predicates. Branches a provider picks
// by key or provider configuration (a Bedrock surface, a Fireworks Anthropic endpoint, a Baseten
// dedicated deployment, a custom provider's allowed requests) are read at their default. nil is a
// provider or branch with no row.
func conversionFor(provider schemas.ModelProvider, model string, kind schemas.RequestType) *conversion {
	chat := kind == schemas.ChatCompletionRequest
	if !chat && kind != schemas.ResponsesRequest {
		return nil
	}
	pick := func(c, r *conversion) *conversion {
		if chat {
			return c
		}
		return r
	}
	switch provider {
	case schemas.Anthropic:
		return pick(&anthropicChat, &anthropicResp)
	case schemas.Azure:
		if schemas.IsAnthropicModelFamily(nil, model) {
			return pick(&hostedClaudeChat, &hostedClaudeResp)
		}
		if !chat && !schemas.AzureServesResponses(schemas.Azure, model) {
			return &responsesViaChat
		}
		return pick(&openAIChat, &openAIResponses)
	case schemas.Vertex:
		switch {
		case schemas.IsAnthropicModelFamily(nil, model):
			return pick(&hostedClaudeChat, &hostedClaudeResp)
		case schemas.IsVertexGeminiModel(nil, model):
			return pick(&geminiChat, &geminiResp)
		}
		return pick(&openAIChat, &responsesViaChat)
	case schemas.GithubCopilot:
		if schemas.IsAnthropicModelFamily(nil, model) {
			return nil
		}
		if !chat && !schemas.GithubCopilotServesResponses(nil, schemas.GithubCopilot, model) {
			return &responsesViaChat
		}
		return pick(&openAIChat, &openAIResponses)
	case schemas.Perplexity:
		if !chat && schemas.PerplexityServesResponses(model) {
			return &openAIResponses
		}
		return pick(&perplexityChat, &perplexityRespViaChat)
	case schemas.Gemini:
		return pick(&geminiChat, &geminiResp)
	case schemas.Bedrock:
		if schemas.IsAnthropicModelFamily(nil, model) {
			return nil
		}
		return pick(&bedrockChat, &bedrockResp)
	case schemas.Cohere:
		return pick(&cohereChat, &cohereResp)
	case schemas.Replicate:
		return pick(&replicateChat, &replicateResp)
	case schemas.HuggingFace:
		return pick(&huggingFaceChat, &responsesViaChat)
	case schemas.Baseten:
		return pick(&basetenChat, &mergingResponses)
	case schemas.BytePlus, schemas.VLLM:
		return pick(&mergingChat, &mergingResponses)
	case schemas.DeepSeek, schemas.MiniMax, schemas.Runware, schemas.Wafer, schemas.SGL:
		return pick(&mergingChat, &responsesViaMergingChat)
	case schemas.OpenAI, schemas.OpenRouter, schemas.XAI, schemas.Fireworks, schemas.OpencodeGo:
		return pick(&openAIChat, &openAIResponses)
	case schemas.Cerebras, schemas.Groq, schemas.Mistral, schemas.Nebius, schemas.Parasail, schemas.Sarvam,
		schemas.Kimchi, schemas.OpencodeZen, schemas.Ollama:
		return pick(&openAIChat, &responsesViaChat)
	}
	return nil
}

// ForwardedRaw is the set of explicit Anthropic features a non-Claude conversion sends upstream
// raw for a chat or Responses request (kind) to provider and model (canonical), with or without
// BifrostContextKeyPassthroughExtraParams. Raw forwarding honours nothing; a Claude conversion
// forwards none raw (it carries what it sends, Uncarried's question). anthropic_beta is never in
// it: the header goes to Claude providers only. Zero for a provider with no row.
func ForwardedRaw(provider schemas.ModelProvider, model string, kind schemas.RequestType, passthroughExtra bool) Feature {
	c := conversionFor(provider, model, kind)
	switch {
	case c == nil:
		return 0
	case passthroughExtra:
		return c.forwardedPassthrough
	}
	return c.forwarded
}

// ConsumedExtraKeys are the top-level ExtraParams keys the conversion for provider and model
// (canonical) reads whether or not passthrough is on; any other key is dropped unless passthrough
// merges it. A conversion that sends every key returns []string{AllExtraKeys}. nil for a provider
// with no row. The slice is shared: callers must not modify it.
func ConsumedExtraKeys(provider schemas.ModelProvider, model string, kind schemas.RequestType) []string {
	if c := conversionFor(provider, model, kind); c != nil {
		return c.consumed
	}
	return nil
}

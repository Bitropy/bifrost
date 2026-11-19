package anthropic

import (
	"context"
	"slices"
	"strings"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
)

// Feature is a set of explicit Anthropic parameters a strip removed because the target provider
// and model cannot carry them: one bit per parameter of a closed vocabulary. The strips OR the bits
// they remove into schemas.BifrostContextKeyRemovedFeatures; in refusal mode
// (schemas.FeatureRefusal) RefuseRemovedFeatures turns a non-empty set into a refusal.
type Feature uint32

// The feature bits, in the order First picks them.
const (
	FeatureContainer Feature = 1 << iota
	FeatureContainerSkills
	FeatureMCPServers
	FeatureSpeed
	FeatureTaskBudget
	FeatureInferenceGeo
	FeatureDiagnostics
	FeatureSafeguards
	FeatureCacheControlScope
	FeatureFallbackCreditToken
	FeatureContextManagement
	FeatureDeferLoading
	FeatureAllowedCallers
	FeatureInputExamples
	FeatureEagerInputStreaming
	FeatureServerTool
	FeatureToolVersion
	FeatureAnthropicBeta
)

var featureNames = [...]string{
	"container", "container_skills", "mcp_servers", "speed", "task_budget", "inference_geo",
	"diagnostics", "safeguards", "cache_control_scope", "fallback_credit_token", "context_management",
	"defer_loading", "allowed_callers", "input_examples", "eager_input_streaming", "server_tool",
	"tool_version", "anthropic_beta",
}

// First returns the name of the lowest set bit, or "" for none.
func (f Feature) First() string {
	for i, name := range featureNames {
		if f&(1<<i) != 0 {
			return name
		}
	}
	return ""
}

// addRemoved ORs f into the request's removed-feature set.
func addRemoved(ctx *schemas.BifrostContext, f Feature) {
	if f == 0 || ctx == nil {
		return
	}
	prev, _ := ctx.Value(schemas.BifrostContextKeyRemovedFeatures).(Feature)
	if prev|f != prev {
		ctx.SetValue(schemas.BifrostContextKeyRemovedFeatures, prev|f)
	}
}

// RemovedFeatures returns the request's removed-feature set.
func RemovedFeatures(ctx context.Context) Feature {
	if bc, isBifrost := ctx.(*schemas.BifrostContext); ctx == nil || (isBifrost && bc == nil) {
		return 0
	}
	f, _ := ctx.Value(schemas.BifrostContextKeyRemovedFeatures).(Feature)
	return f
}

// addAdaptation is schemas.AddAdaptation, tolerating a nil context.
func addAdaptation(ctx *schemas.BifrostContext, a schemas.Adaptation) {
	if ctx != nil {
		schemas.AddAdaptation(ctx, a)
	}
}

// RefuseRemovedFeatures is the one refusal check: in refusal mode it returns the refusal for the
// first feature a strip removed from the request on ctx, and nil otherwise. Callers run it once the
// request is built, before any network I/O.
func RefuseRemovedFeatures(ctx *schemas.BifrostContext, provider schemas.ModelProvider) *schemas.UnsupportedOnTargetError {
	if !schemas.FeatureRefusal(ctx) {
		return nil
	}
	if removed := RemovedFeatures(ctx); removed != 0 {
		return &schemas.UnsupportedOnTargetError{Feature: removed.First(), Provider: provider}
	}
	return nil
}

// refusalError is RefuseRemovedFeatures as the error a request builder returns.
func refusalError(ctx *schemas.BifrostContext, provider schemas.ModelProvider) error {
	if refusal := RefuseRemovedFeatures(ctx, provider); refusal != nil {
		return refusal
	}
	return nil
}

// reportClientBetas records anthropic_beta when a client-sourced anthropic-beta value
// (schemas.BifrostContextKeyClientAnthropicBeta) is one FilterBetaHeadersForProvider drops for
// provider. Betas from the provider's configuration are the operator's and are not reported.
func reportClientBetas(ctx *schemas.BifrostContext, provider schemas.ModelProvider, overrides map[string]bool) {
	if ctx == nil {
		return
	}
	values, _ := ctx.Value(schemas.BifrostContextKeyClientAnthropicBeta).([]string)
	if len(values) == 0 {
		return
	}
	merged := mergeBetaValues(nil, slices.Values(values))
	if len(FilterBetaHeadersForProvider(merged, provider, overrides)) < len(merged) {
		addRemoved(ctx, FeatureAnthropicBeta)
	}
}

// stripMergedBody runs in refusal mode on a typed build's body after the passthrough merge, so a
// raw key the merge added is gated like a typed one: the raw strip (reporting what it removes and
// adapts), then the raw model gate for top_k on an adaptive-only model.
func stripMergedBody(ctx *schemas.BifrostContext, jsonBody []byte, provider schemas.ModelProvider, model string) ([]byte, error) {
	jsonBody, err := StripUnsupportedFieldsFromRawBodyReport(ctx, jsonBody, provider, model)
	if err != nil {
		return nil, err
	}
	if providerUtils.JSONFieldExists(jsonBody, "top_k") &&
		schemas.ResolveModelCaps(provider, model).AdaptiveOnlyThinking(DefaultAdaptiveOnlyThinking(model)) {
		if jsonBody, err = providerUtils.DeleteJSONField(jsonBody, "top_k"); err != nil {
			return nil, err
		}
		addAdaptation(ctx, schemas.AdaptationTopK)
	}
	return jsonBody, nil
}

// noteIgnoredChatParams records unmapped when the request sets a chat parameter
// ToAnthropicChatRequest has no mapping for, so it is ignored.
func noteIgnoredChatParams(ctx *schemas.BifrostContext, p *schemas.ChatParameters) {
	if p == nil {
		return
	}
	if (p.N != nil && *p.N > 1) || (p.LogProbs != nil && *p.LogProbs) || p.TopLogProbs != nil || p.Seed != nil ||
		p.PresencePenalty != nil || p.FrequencyPenalty != nil || p.LogitBias != nil || p.Prediction != nil ||
		p.Audio != nil || slices.ContainsFunc(p.Modalities, func(m string) bool { return !strings.EqualFold(m, "text") }) ||
		p.WebSearchOptions != nil || p.Verbosity != nil {
		addAdaptation(ctx, schemas.AdaptationUnmapped)
	}
}

// noteIgnoredResponsesParams records unmapped when the request sets a Responses parameter
// ToAnthropicResponsesRequest does not read and that changes the answer. Members the embedder
// refuses (background, truncation "auto", max_tool_calls) and members that change no answer
// (include, metadata, store, stream_options, prompt_cache_key, prompt_cache_retention,
// safety_identifier) are not listed.
func noteIgnoredResponsesParams(ctx *schemas.BifrostContext, p *schemas.ResponsesParameters) {
	if p == nil {
		return
	}
	if p.Conversation != nil || p.PreviousResponseID != nil || p.PromptCacheOptions != nil || p.TopLogProbs != nil ||
		p.IncludeServerSideToolInvocations != nil || (p.Text != nil && p.Text.Verbosity != nil) {
		addAdaptation(ctx, schemas.AdaptationUnmapped)
	}
}

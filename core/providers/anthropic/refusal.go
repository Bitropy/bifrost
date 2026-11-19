package anthropic

import (
	"context"
	"slices"
	"strings"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/tidwall/gjson"
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

// reportRawRemoved records f before the raw strip deletes path from jsonBody, unless the value
// there carries nothing (null, "", [] or {}): deleting an empty value removes nothing the client
// asked for, as the typed strip, which only sees non-empty values, agrees.
func reportRawRemoved(ctx *schemas.BifrostContext, jsonBody []byte, path string, f Feature) {
	if ctx != nil && rawMeaningful(providerUtils.GetJSONField(jsonBody, path)) {
		addRemoved(ctx, f)
	}
}

// rawMeaningful reports whether v exists and is not null, "", [] or {}.
func rawMeaningful(v gjson.Result) bool {
	switch {
	case !v.Exists() || v.Type == gjson.Null:
		return false
	case v.Type == gjson.String:
		return v.Str != ""
	case v.IsArray() || v.IsObject():
		empty := true
		v.ForEach(func(_, _ gjson.Result) bool {
			empty = false
			return false
		})
		return !empty
	}
	return true
}

// RemovedFeatures returns the request's removed-feature set.
func RemovedFeatures(ctx context.Context) Feature {
	if bc, isBifrost := ctx.(*schemas.BifrostContext); ctx == nil || (isBifrost && bc == nil) {
		return 0
	}
	f, _ := ctx.Value(schemas.BifrostContextKeyRemovedFeatures).(Feature)
	return f
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

// reportClientBetas records anthropic_beta, in refusal mode, when a client-sourced anthropic-beta value
// (schemas.BifrostContextKeyClientAnthropicBeta) is one FilterBetaHeadersForProvider drops for
// provider. Betas from the provider's configuration are the operator's and are not reported.
func reportClientBetas(ctx *schemas.BifrostContext, provider schemas.ModelProvider, overrides map[string]bool) {
	if !schemas.FeatureRefusal(ctx) {
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
		schemas.AddAdaptation(ctx, schemas.AdaptationTopK)
	}
	return jsonBody, nil
}

// IgnoredParam is a generic parameter a converter has no mapping for: Name is its wire name, Set
// reports whether the request carries it.
type IgnoredParam[P any] struct {
	Name string
	Set  func(p *P) bool
}

// IgnoredChatParams are the chat parameters ToAnthropicChatRequest ignores.
var IgnoredChatParams = []IgnoredParam[schemas.ChatParameters]{
	{"n", func(p *schemas.ChatParameters) bool { return p.N != nil && *p.N > 1 }},
	{"logprobs", func(p *schemas.ChatParameters) bool { return p.LogProbs != nil && *p.LogProbs }},
	{"top_logprobs", func(p *schemas.ChatParameters) bool { return p.TopLogProbs != nil }},
	{"seed", func(p *schemas.ChatParameters) bool { return p.Seed != nil }},
	{"presence_penalty", func(p *schemas.ChatParameters) bool { return p.PresencePenalty != nil }},
	{"frequency_penalty", func(p *schemas.ChatParameters) bool { return p.FrequencyPenalty != nil }},
	{"logit_bias", func(p *schemas.ChatParameters) bool { return p.LogitBias != nil }},
	{"prediction", func(p *schemas.ChatParameters) bool { return p.Prediction != nil }},
	{"modalities", func(p *schemas.ChatParameters) bool {
		return slices.ContainsFunc(p.Modalities, func(m string) bool { return !strings.EqualFold(m, "text") })
	}},
	{"audio", func(p *schemas.ChatParameters) bool { return p.Audio != nil }},
	{"web_search_options", func(p *schemas.ChatParameters) bool { return p.WebSearchOptions != nil }},
	{"verbosity", func(p *schemas.ChatParameters) bool { return p.Verbosity != nil }},
}

// IgnoredResponsesParams are the Responses parameters ToAnthropicResponsesRequest does not read and
// that change the answer. Members the embedder refuses (background, truncation "auto",
// max_tool_calls) and members that change no answer (include, metadata, store, stream_options,
// prompt_cache_key, prompt_cache_retention, safety_identifier) are not listed.
var IgnoredResponsesParams = []IgnoredParam[schemas.ResponsesParameters]{
	{"conversation", func(p *schemas.ResponsesParameters) bool { return p.Conversation != nil }},
	{"previous_response_id", func(p *schemas.ResponsesParameters) bool { return p.PreviousResponseID != nil }},
	{"prompt_cache_options", func(p *schemas.ResponsesParameters) bool { return p.PromptCacheOptions != nil }},
	{"top_logprobs", func(p *schemas.ResponsesParameters) bool { return p.TopLogProbs != nil }},
	{"include_server_side_tool_invocations", func(p *schemas.ResponsesParameters) bool {
		return p.IncludeServerSideToolInvocations != nil
	}},
	{"text.verbosity", func(p *schemas.ResponsesParameters) bool { return p.Text != nil && p.Text.Verbosity != nil }},
}

// noteIgnored records unmapped when the request sets a parameter of ignored.
func noteIgnored[P any](ctx *schemas.BifrostContext, p *P, ignored []IgnoredParam[P]) {
	if p == nil {
		return
	}
	for _, param := range ignored {
		if param.Set(p) {
			schemas.AddAdaptation(ctx, schemas.AdaptationUnmapped)
			return
		}
	}
}

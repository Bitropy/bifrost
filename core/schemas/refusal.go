package schemas

import (
	"context"
	"io"
	"strings"
)

// Adaptation is a set of generic request parameters core (or the embedder) changed so the target
// model could take the request: one bit per adaptation of a closed vocabulary. The zero value is
// "none".
type Adaptation uint32

// The adaptation bits. Their order is the order String writes them in.
const (
	AdaptationSampling Adaptation = 1 << iota
	AdaptationToolChoice
	AdaptationMaxTokens
	AdaptationReasoning
	AdaptationServiceTier
	AdaptationStrict
	AdaptationUnmapped
	AdaptationTopK
	AdaptationPrefill
	// AdaptationToolSchema: an empty root anyOf/oneOf/allOf was dropped from a tool schema.
	AdaptationToolSchema
	// AdaptationVoice: an OpenAI voice name was mapped to the target's own voice.
	AdaptationVoice
)

var adaptationNames = [...]string{"sampling", "tool_choice", "max_tokens", "reasoning", "service_tier", "strict", "unmapped", "top_k", "prefill", "tool_schema", "voice"}

// Has reports whether every bit of b is set in a.
func (a Adaptation) Has(b Adaptation) bool { return a&b == b }

// Add returns a with the bits of b set.
func (a Adaptation) Add(b Adaptation) Adaptation { return a | b }

// String joins the names of the set bits in bit order with ", ", or returns "" for none.
func (a Adaptation) String() string {
	if a == 0 {
		return ""
	}
	var sb strings.Builder
	for i, name := range adaptationNames {
		if a&(1<<i) == 0 {
			continue
		}
		if sb.Len() > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString(name)
	}
	return sb.String()
}

// AddAdaptation ORs a into the attempt's adaptation set on ctx. The set starts from the value the
// embedder left on a parent context, so the embedder's own adaptations and core's are one set.
// Nothing is written when a is zero.
func AddAdaptation(ctx *BifrostContext, a Adaptation) {
	if a == 0 || ctx == nil {
		return
	}
	ctx.updateValue(BifrostContextKeyAdaptations, func(current any) any {
		prev, _ := current.(Adaptation)
		return prev | a
	})
}

// ResetAdaptations drops the adaptations core recorded on ctx and keeps the set the embedder left on
// a parent context: core's describe one provider attempt, the embedder's edits hold for the whole call.
func ResetAdaptations(ctx *BifrostContext) {
	if ctx != nil {
		ctx.deleteValue(BifrostContextKeyAdaptations)
	}
}

// AdaptationsFromContext returns the attempt's adaptation set.
func AdaptationsFromContext(ctx context.Context) Adaptation {
	if bc, isBifrost := ctx.(*BifrostContext); ctx == nil || (isBifrost && bc == nil) {
		return 0
	}
	a, _ := ctx.Value(BifrostContextKeyAdaptations).(Adaptation)
	return a
}

// AnthropicFeature is a set of explicit Anthropic parameters the target provider and model cannot
// carry: one bit per parameter of a closed vocabulary. It lives here, not in the anthropic
// provider, because RequestBodyRewriteHook carries a mask of it; providers/anthropic aliases it
// as Feature and names the bits.
type AnthropicFeature uint32

var anthropicFeatureNames = [...]string{
	"container", "container_skills", "mcp_servers", "speed", "task_budget", "inference_geo",
	"diagnostics", "safeguards", "cache_control_scope", "fallback_credit_token", "context_management",
	"defer_loading", "allowed_callers", "input_examples", "eager_input_streaming", "server_tool",
	"tool_version", "anthropic_beta", "fallback_block", "compaction_block", "tool_change_block",
}

// First returns the name of the lowest set bit, or "" for none.
func (f AnthropicFeature) First() string {
	for i, name := range anthropicFeatureNames {
		if f&(1<<i) != 0 {
			return name
		}
	}
	return ""
}

// String joins the names of the set bits in bit order with ", ", or returns "" for none. A bit
// past the vocabulary has no name and is not written.
func (f AnthropicFeature) String() string {
	var sb strings.Builder
	for i, name := range anthropicFeatureNames {
		if f&(1<<i) == 0 {
			continue
		}
		if sb.Len() > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString(name)
	}
	return sb.String()
}

// RequestBodyRewriteHook is the per-call request policy an embedder installs under
// BifrostContextKeyRequestBodyRewriter, as a pointer (a pointer fits the context value without an
// allocation). Rewrite is the request-body rewriter (providerUtils.RequestBodyRewriter); nil
// rewrites nothing. RefuseMask is the set of explicit Anthropic parameters an Anthropic-family
// request builder refuses (*UnsupportedOnTargetError) instead of removing when the target cannot
// carry them; every other removed parameter is stripped and reported. Zero is upstream behaviour:
// silent strips.
type RequestBodyRewriteHook struct {
	Rewrite    func(body []byte) (r io.Reader, size int64, err error)
	RefuseMask AnthropicFeature
}

// RefuseMaskFromContext returns the refuse mask of the hook on ctx, zero without one.
func RefuseMaskFromContext(ctx context.Context) AnthropicFeature {
	if bc, isBifrost := ctx.(*BifrostContext); ctx == nil || (isBifrost && bc == nil) {
		return 0
	}
	h, _ := ctx.Value(BifrostContextKeyRequestBodyRewriter).(*RequestBodyRewriteHook)
	if h == nil {
		return 0
	}
	return h.RefuseMask
}

// AdaptedFeaturesFromContext returns the attempt's removed features outside the refuse mask: what
// it stripped and serves without. A feature core reports but still forwards must be in every mask
// an embedder installs, so it never shows here as stripped.
func AdaptedFeaturesFromContext(ctx context.Context) AnthropicFeature {
	if bc, isBifrost := ctx.(*BifrostContext); ctx == nil || (isBifrost && bc == nil) {
		return 0
	}
	removed, _ := ctx.Value(BifrostContextKeyRemovedFeatures).(AnthropicFeature)
	if removed == 0 {
		return 0
	}
	return removed &^ RefuseMaskFromContext(ctx)
}

// UnsupportedOnTargetError is the refusal a request builder returns in refusal mode, before any
// network I/O, for an explicit parameter the target provider and model cannot carry. Feature is
// the parameter's name in a closed vocabulary; it never holds client text. It is carried on the
// BifrostError's ErrorField.Error, so errors.As finds it at the embedder.
type UnsupportedOnTargetError struct {
	Feature  string
	Provider ModelProvider
}

func (e *UnsupportedOnTargetError) Error() string {
	return e.Feature + " is not supported on provider " + string(e.Provider)
}

// ErrorCodeUnsupportedOnTarget is ErrorField.Code of an UnsupportedOnTargetError.
const ErrorCodeUnsupportedOnTarget = "unsupported_on_target"

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
)

var adaptationNames = [...]string{"sampling", "tool_choice", "max_tokens", "reasoning", "service_tier", "strict", "unmapped", "top_k", "prefill", "tool_schema"}

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

// RequestBodyRewriteHook is the per-call request policy an embedder installs under
// BifrostContextKeyRequestBodyRewriter, as a pointer (a pointer fits the context value without an
// allocation). Rewrite is the request-body rewriter (providerUtils.RequestBodyRewriter); nil
// rewrites nothing. RefuseFeatures turns on refusal mode: an Anthropic-family request builder
// returns *UnsupportedOnTargetError for an explicit Anthropic parameter the target cannot carry,
// instead of removing it.
type RequestBodyRewriteHook struct {
	Rewrite        func(body []byte) (r io.Reader, size int64, err error)
	RefuseFeatures bool
}

// FeatureRefusal reports whether the call on ctx runs in refusal mode.
func FeatureRefusal(ctx context.Context) bool {
	if bc, isBifrost := ctx.(*BifrostContext); ctx == nil || (isBifrost && bc == nil) {
		return false
	}
	h, _ := ctx.Value(BifrostContextKeyRequestBodyRewriter).(*RequestBodyRewriteHook)
	return h != nil && h.RefuseFeatures
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

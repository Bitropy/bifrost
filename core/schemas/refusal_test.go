package schemas

import (
	"context"
	"io"
	"testing"
)

func TestAdaptation_StringHasAdd(t *testing.T) {
	if got := Adaptation(0).String(); got != "" {
		t.Fatalf("none = %q", got)
	}
	all := Adaptation(0)
	for i := range adaptationNames {
		all = all.Add(1 << i)
	}
	if got, want := all.String(), "sampling, tool_choice, max_tokens, reasoning, service_tier, strict, unmapped, top_k, prefill, tool_schema, voice"; got != want {
		t.Fatalf("all = %q, want %q", got, want)
	}
	a := AdaptationTopK.Add(AdaptationSampling)
	if got := a.String(); got != "sampling, top_k" {
		t.Fatalf("bit order: %q", got)
	}
	if !a.Has(AdaptationSampling) || a.Has(AdaptationReasoning) || !a.Has(AdaptationSampling|AdaptationTopK) {
		t.Fatal("Has")
	}
}

// AddAdaptation starts from the embedder's bits on a parent context and writes nothing for zero.
func TestAddAdaptation_StartsFromParent(t *testing.T) {
	parent := context.WithValue(context.Background(), BifrostContextKeyAdaptations, AdaptationReasoning)
	ctx := NewBifrostContext(parent, NoDeadline)
	AddAdaptation(ctx, 0)
	if ctx.userValues[BifrostContextKeyAdaptations] != nil {
		t.Fatal("zero wrote a value")
	}
	AddAdaptation(ctx, AdaptationSampling)
	if got := AdaptationsFromContext(ctx); got != AdaptationReasoning|AdaptationSampling {
		t.Fatalf("got %v", got)
	}
	if AdaptationsFromContext(nil) != 0 || AdaptationsFromContext((*BifrostContext)(nil)) != 0 {
		t.Fatal("nil context")
	}
}

func TestRefuseMaskFromContext(t *testing.T) {
	if RefuseMaskFromContext(nil) != 0 || RefuseMaskFromContext((*BifrostContext)(nil)) != 0 || RefuseMaskFromContext(context.Background()) != 0 {
		t.Fatal("no hook is zero")
	}
	ctx := NewBifrostContext(context.Background(), NoDeadline)
	ctx.SetValue(BifrostContextKeyRequestBodyRewriter, &RequestBodyRewriteHook{RefuseMask: 1<<5 | 1<<12})
	if got := RefuseMaskFromContext(ctx); got != 1<<5|1<<12 {
		t.Fatalf("mask = %v", got)
	}
	ctx.SetValue(BifrostContextKeyRequestBodyRewriter, (*RequestBodyRewriteHook)(nil))
	if RefuseMaskFromContext(ctx) != 0 {
		t.Fatal("nil hook is zero")
	}
}

// The adapted set is what the strips removed outside the refuse mask, read through a parent value.
func TestAdaptedFeaturesFromContext(t *testing.T) {
	if AdaptedFeaturesFromContext(nil) != 0 || AdaptedFeaturesFromContext((*BifrostContext)(nil)) != 0 {
		t.Fatal("nil context")
	}
	parent := context.WithValue(context.Background(), BifrostContextKeyRemovedFeatures, AnthropicFeature(1<<3|1<<5))
	ctx := NewBifrostContext(parent, NoDeadline)
	if got := AdaptedFeaturesFromContext(ctx); got != 1<<3|1<<5 {
		t.Fatalf("no hook: %v", got)
	}
	ctx.SetValue(BifrostContextKeyRequestBodyRewriter, &RequestBodyRewriteHook{RefuseMask: 1 << 5})
	if got := AdaptedFeaturesFromContext(ctx); got != 1<<3 {
		t.Fatalf("masked: %v", got)
	}
}

func TestAnthropicFeature_FirstString(t *testing.T) {
	var names []string
	for i := range 32 {
		if name := AnthropicFeature(1 << i).First(); name != "" {
			names = append(names, name)
		}
	}
	if len(names) != 21 {
		t.Fatalf("names = %v", names)
	}
	if got := AnthropicFeature(0).String(); got != "" {
		t.Fatalf("none = %q", got)
	}
	if got, want := AnthropicFeature(1<<17|1<<3|1<<31).String(), "speed, anthropic_beta"; got != want {
		t.Fatalf("String = %q, want %q", got, want)
	}
	if got := AnthropicFeature(1<<17 | 1<<3).First(); got != "speed" {
		t.Fatalf("First = %q", got)
	}
}

// A hook without a rewrite sizes tool-result media like no rewriter at all.
func TestToolResultMediaSized_Hook(t *testing.T) {
	ctx := NewBifrostContext(context.Background(), NoDeadline)
	ctx.SetValue(BifrostContextKeyRequestBodyRewriter, &RequestBodyRewriteHook{RefuseMask: 1})
	if !toolResultMediaSized(ctx) {
		t.Fatal("nil Rewrite: sized")
	}
	ctx.SetValue(BifrostContextKeyRequestBodyRewriter, &RequestBodyRewriteHook{Rewrite: func([]byte) (io.Reader, int64, error) { return nil, 0, nil }})
	if toolResultMediaSized(ctx) {
		t.Fatal("rewrite: not sized")
	}
}

func TestRefuseMask_AllocBudget(t *testing.T) {
	ctx := NewBifrostContext(context.Background(), NoDeadline)
	ctx.SetValue(BifrostContextKeyRequestBodyRewriter, &RequestBodyRewriteHook{RefuseMask: 1})
	if allocs := testing.AllocsPerRun(100, func() {
		_ = RefuseMaskFromContext(ctx)
		_ = AdaptationsFromContext(ctx)
		_ = AdaptedFeaturesFromContext(ctx)
	}); allocs != 0 {
		t.Fatalf("allocs = %v", allocs)
	}
}

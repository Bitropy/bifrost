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
	if got, want := all.String(), "sampling, tool_choice, max_tokens, reasoning, service_tier, strict, unmapped, top_k, prefill"; got != want {
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

func TestFeatureRefusal(t *testing.T) {
	if FeatureRefusal(nil) || FeatureRefusal((*BifrostContext)(nil)) || FeatureRefusal(context.Background()) {
		t.Fatal("no hook is off")
	}
	ctx := NewBifrostContext(context.Background(), NoDeadline)
	ctx.SetValue(BifrostContextKeyRequestBodyRewriter, &RequestBodyRewriteHook{RefuseFeatures: true})
	if !FeatureRefusal(ctx) {
		t.Fatal("hook on")
	}
	ctx.SetValue(BifrostContextKeyRequestBodyRewriter, (*RequestBodyRewriteHook)(nil))
	if FeatureRefusal(ctx) {
		t.Fatal("nil hook is off")
	}
}

// A hook without a rewrite sizes tool-result media like no rewriter at all.
func TestToolResultMediaSized_Hook(t *testing.T) {
	ctx := NewBifrostContext(context.Background(), NoDeadline)
	ctx.SetValue(BifrostContextKeyRequestBodyRewriter, &RequestBodyRewriteHook{RefuseFeatures: true})
	if !toolResultMediaSized(ctx) {
		t.Fatal("nil Rewrite: sized")
	}
	ctx.SetValue(BifrostContextKeyRequestBodyRewriter, &RequestBodyRewriteHook{Rewrite: func([]byte) (io.Reader, int64, error) { return nil, 0, nil }})
	if toolResultMediaSized(ctx) {
		t.Fatal("rewrite: not sized")
	}
}

func TestFeatureRefusal_AllocBudget(t *testing.T) {
	ctx := NewBifrostContext(context.Background(), NoDeadline)
	ctx.SetValue(BifrostContextKeyRequestBodyRewriter, &RequestBodyRewriteHook{RefuseFeatures: true})
	if allocs := testing.AllocsPerRun(100, func() { _ = FeatureRefusal(ctx); _ = AdaptationsFromContext(ctx) }); allocs != 0 {
		t.Fatalf("allocs = %v", allocs)
	}
}

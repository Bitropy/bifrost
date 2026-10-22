package gemini

import (
	"math"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

// A chat request's seed reaches Gemini as generationConfig.seed; it used to be
// dropped, so a client asking for reproducible sampling silently did not get
// it (BIT-2741 G1). Gemini's seed is an int32: a seed outside that range is
// left out as before rather than wrapped (a different seed) or refused (a
// request that used to be served).
func TestConvertParamsToGenerationConfig_MapsSeed(t *testing.T) {
	config, err := convertParamsToGenerationConfig(&schemas.ChatParameters{Seed: schemas.Ptr(42)}, nil, schemas.Gemini, "gemini-2.5-flash")
	if err != nil {
		t.Fatal(err)
	}
	if config.Seed == nil || *config.Seed != 42 {
		t.Fatalf("generationConfig.seed = %v, want 42", config.Seed)
	}

	config, err = convertParamsToGenerationConfig(&schemas.ChatParameters{}, nil, schemas.Gemini, "gemini-2.5-flash")
	if err != nil || config.Seed != nil {
		t.Fatalf("no seed asked: seed = %v, err = %v", config.Seed, err)
	}

	for _, seed := range []int{math.MaxInt32 + 1, math.MinInt32 - 1} {
		config, err = convertParamsToGenerationConfig(&schemas.ChatParameters{Seed: schemas.Ptr(seed)}, nil, schemas.Gemini, "gemini-2.5-flash")
		if err != nil || config.Seed != nil {
			t.Fatalf("seed %d outside int32: seed = %v, err = %v; want left out", seed, config.Seed, err)
		}
	}
}

package anthropic

import (
	"context"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// AC-3: a Responses web tool whose client-selected version the host remaps is sent downgraded and
// recorded as tool_version at the tool actually sent; a host without the remap records nothing.
// webToolsSupplyCodeExecution resolves the same versions but takes no context, so it cannot record.
func TestConvertBifrostToolToAnthropic_ReportsToolVersion(t *testing.T) {
	webSearch := func(version string) schemas.ResponsesTool {
		return schemas.ResponsesTool{Type: schemas.ResponsesToolTypeWebSearch, ResponsesToolWebSearch: &schemas.ResponsesToolWebSearch{Version: schemas.Ptr(version)}}
	}
	cases := []struct {
		name     string
		provider schemas.ModelProvider
		version  string
		wantType AnthropicToolType
		want     Feature
	}{
		{"vertex remaps 20260209", schemas.Vertex, string(AnthropicToolTypeWebSearch20260209), AnthropicToolTypeWebSearch20250305, FeatureToolVersion},
		{"vertex keeps 20250305", schemas.Vertex, string(AnthropicToolTypeWebSearch20250305), AnthropicToolTypeWebSearch20250305, 0},
		{"anthropic keeps 20260209", schemas.Anthropic, string(AnthropicToolTypeWebSearch20260209), AnthropicToolTypeWebSearch20260209, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			caps := schemas.ResolveModelCaps(tc.provider, "claude-sonnet-4-6")
			tools := []schemas.ResponsesTool{webSearch(tc.version), webSearch(tc.version)}

			ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
			out := convertBifrostToolToAnthropic(ctx, caps, &tools[0], tc.provider, false)
			require.NotNil(t, out)
			require.NotNil(t, out.Type)
			assert.Equal(t, tc.wantType, *out.Type)
			assert.Equal(t, tc.want, RemovedFeatures(ctx))

			ctx = schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
			converted, _, err := convertBifrostToolsToAnthropic(ctx, caps, tools, tc.provider)
			require.NoError(t, err)
			require.Len(t, converted, 2)
			assert.Equal(t, tc.want, RemovedFeatures(ctx), "two remapped tools record the one bit")
		})
	}
}

package schemas

import (
	"context"
	"testing"
	"time"

	"github.com/bytedance/sonic"
	"github.com/stretchr/testify/require"
)

// TestReportChatCacheControlDropped: a converter with no cache breakpoint reports a
// marker it drops, and nothing for an absent or null one.
func TestReportChatCacheControlDropped(t *testing.T) {
	eph := &CacheControl{Type: CacheControlTypeEphemeral}
	for _, row := range []struct {
		name, msgs string
		tools      []ChatTool
		want       bool
	}{
		{name: "message marker", msgs: `[{"role":"user","content":"x","cache_control":{"type":"ephemeral"}}]`, want: true},
		{name: "tool marker", msgs: `[{"role":"user","content":"x"}]`, tools: []ChatTool{{Type: ChatToolTypeFunction, Function: &ChatToolFunction{Name: "f"}, CacheControl: eph}}, want: true},
		{name: "null marker", msgs: `[{"role":"user","content":"x","cache_control":null}]`},
		{name: "part null marker", msgs: `[{"role":"user","content":[{"type":"text","text":"x","cache_control":null}]}]`},
		{name: "no marker", msgs: `[{"role":"user","content":"x"}]`},
	} {
		t.Run(row.name, func(t *testing.T) {
			var msgs []ChatMessage
			require.NoError(t, sonic.Unmarshal([]byte(row.msgs), &msgs))
			ctx := NewBifrostContext(context.Background(), time.Time{})
			t.Cleanup(ctx.Cancel)
			ReportChatCacheControlDropped(ctx, &BifrostChatRequest{Input: msgs, Params: &ChatParameters{Tools: row.tools}})
			require.Equal(t, row.want, AdaptationsFromContext(ctx).Has(AdaptationUnmapped))
		})
	}
}

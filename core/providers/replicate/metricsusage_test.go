package replicate_test

import (
	"encoding/json"
	"testing"

	"github.com/maximhq/bifrost/core/providers/replicate"
	"github.com/stretchr/testify/require"
)

// TestPredictionUsage_PrefersMetricsTokenCounts: a succeeded prediction that
// reports its token counts in metrics but not in its logs bills those counts
// on chat, responses and text (BIT-2752). Before, usage came from the log
// lines only, so empty logs billed nothing.
func TestPredictionUsage_PrefersMetricsTokenCounts(t *testing.T) {
	in, out := 16, 8
	empty := ""
	logs := "Input token count: 3\nOutput token count: 2\n"
	for _, tc := range []struct {
		name    string
		logs    *string
		metrics *replicate.ReplicateMetrics
		in, out int
	}{
		{"metrics, empty logs", &empty, &replicate.ReplicateMetrics{InputTokenCount: &in, OutputTokenCount: &out}, 16, 8},
		{"metrics, no logs", nil, &replicate.ReplicateMetrics{InputTokenCount: &in, OutputTokenCount: &out}, 16, 8},
		{"metrics win over logs", &logs, &replicate.ReplicateMetrics{InputTokenCount: &in, OutputTokenCount: &out}, 16, 8},
		{"logs without metrics counts", &logs, &replicate.ReplicateMetrics{}, 3, 2},
		{"output from metrics, input from logs", &logs, &replicate.ReplicateMetrics{OutputTokenCount: &out}, 3, 8},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := &replicate.ReplicatePredictionResponse{
				Status:  replicate.ReplicatePredictionStatusSucceeded,
				Logs:    tc.logs,
				Metrics: tc.metrics,
			}

			chat := p.ToBifrostChatResponse()
			require.NotNil(t, chat.Usage)
			require.Equal(t, tc.in, chat.Usage.PromptTokens)
			require.Equal(t, tc.out, chat.Usage.CompletionTokens)
			require.Equal(t, tc.in+tc.out, chat.Usage.TotalTokens)

			resp := p.ToBifrostResponsesResponse()
			require.NotNil(t, resp.Usage)
			require.Equal(t, tc.in, resp.Usage.InputTokens)
			require.Equal(t, tc.out, resp.Usage.OutputTokens)
			require.Equal(t, tc.in+tc.out, resp.Usage.TotalTokens)

			text := p.ToBifrostTextCompletionResponse()
			require.NotNil(t, text.Usage)
			require.Equal(t, tc.in, text.Usage.PromptTokens)
			require.Equal(t, tc.out, text.Usage.CompletionTokens)
			require.Equal(t, tc.in+tc.out, text.Usage.TotalTokens)
		})
	}
}

// TestPredictionUsage_UnmarshalsMetricsTokenCounts pins the wire names.
func TestPredictionUsage_UnmarshalsMetricsTokenCounts(t *testing.T) {
	var m replicate.ReplicateMetrics
	require.NoError(t, json.Unmarshal([]byte(`{"input_token_count":16,"output_token_count":8,"predict_time":0.4}`), &m))
	require.NotNil(t, m.InputTokenCount)
	require.NotNil(t, m.OutputTokenCount)
	require.Equal(t, 16, *m.InputTokenCount)
	require.Equal(t, 8, *m.OutputTokenCount)
}

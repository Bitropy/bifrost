package anthropic

import (
	"runtime/debug"
	"strconv"
	"testing"

	"github.com/bytedance/sonic"
)

// textStreamEvents is a plain text answer of about 500 events: no provider block,
// no citation, the shape almost every chat stream has.
func textStreamEvents(tb testing.TB, deltas int) []AnthropicStreamEvent {
	tb.Helper()
	raw := []string{
		`{"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","model":"claude-sonnet-4-5","content":[],"usage":{"input_tokens":12,"output_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
	}
	for i := 0; i < deltas; i++ {
		raw = append(raw, `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"word`+strconv.Itoa(i)+` "}}`)
	}
	raw = append(raw,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":500}}`,
		`{"type":"message_stop"}`,
	)
	events := make([]AnthropicStreamEvent, len(raw))
	for i, s := range raw {
		if err := sonic.Unmarshal([]byte(s), &events[i]); err != nil {
			tb.Fatal(err)
		}
	}
	return events
}

// convertAndMarshalStream converts every event and marshals each chunk's delta, the
// per-event work a chat stream pays between the provider and the client socket.
func convertAndMarshalStream(events []AnthropicStreamEvent) {
	state := NewAnthropicStreamState()
	for i := range events {
		resp, _, _ := events[i].ToBifrostChatCompletionStream(nil, "", state)
		if resp == nil {
			continue
		}
		for _, c := range resp.Choices {
			if c.ChatStreamResponseChoice != nil && c.ChatStreamResponseChoice.Delta != nil {
				_, _ = sonic.Marshal(c.ChatStreamResponseChoice.Delta)
			}
		}
	}
}

// Measured on the converter before provider blocks existed; a stream without one must
// not pay for them: the same allocations and at most 1% more bytes.
const (
	textStreamDeltas     = 495
	textStreamAllocs     = 4465
	textStreamBytes      = 623083
	textStreamBytesSlack = textStreamBytes / 100
)

func TestAnthropicChatStream_NoProviderBlock_BytesBudget(t *testing.T) {
	skipAllocUnderRace(t)
	// A collection mid-run empties the sync.Pools the converter draws from, and the refill
	// would count as the converter's bytes.
	gcPercent := debug.SetGCPercent(-1)
	t.Cleanup(func() { debug.SetGCPercent(gcPercent) })
	events := textStreamEvents(t, textStreamDeltas)
	convertAndMarshalStream(events) // warm sonic's encoder cache
	allocs := testing.AllocsPerRun(50, func() { convertAndMarshalStream(events) })
	res := testing.Benchmark(func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			convertAndMarshalStream(events)
		}
	})
	bytes := res.AllocedBytesPerOp()
	t.Logf("text stream of %d events: %v allocs, %d B", len(events), allocs, bytes)
	if allocs > textStreamAllocs {
		t.Errorf("allocates %v per stream, budget %d", allocs, textStreamAllocs)
	}
	if bytes > textStreamBytes+textStreamBytesSlack {
		t.Errorf("allocates %d B per stream, budget %d (+%d)", bytes, textStreamBytes, textStreamBytesSlack)
	}
}

// BenchmarkAnthropicTextStreamConvertAndMarshal is the budget test's workload as a
// benchmark, for A/B runs of the per-event cost.
func BenchmarkAnthropicTextStreamConvertAndMarshal(b *testing.B) {
	events := textStreamEvents(b, textStreamDeltas)
	b.ReportAllocs()
	for b.Loop() {
		convertAndMarshalStream(events)
	}
}

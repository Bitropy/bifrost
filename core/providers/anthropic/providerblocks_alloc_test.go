package anthropic

import (
	"strings"
	"testing"

	"github.com/bytedance/sonic"
)

// The no-provider-block budgets are the counts these fixtures allocate on the
// base pin: provider blocks may cost nothing on an answer that has none.
const (
	noProviderBlockStreamAllocs = 27
	noProviderBlockUnaryAllocs  = 24
	// providerBlockStreamCeiling is what one unmapped block allocates on a warm
	// stream: the chunk's response, choice slice, stream choice, delta, one-entry
	// slice and the block copy.
	providerBlockStreamCeiling = 6
)

var noProviderBlockStreamEvents = []string{
	`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
	`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"let me think"}}`,
	`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}`,
	`{"type":"content_block_stop","index":0}`,
	`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
	`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hello there."}}`,
	`{"type":"content_block_stop","index":1}`,
	`{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_1","name":"f","input":{}}}`,
	`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"a\":1}"}}`,
	`{"type":"content_block_stop","index":2}`,
}

const noProviderBlockUnary = `{"id":"m","type":"message","role":"assistant","model":"claude","stop_reason":"tool_use","content":[{"type":"thinking","thinking":"hmm","signature":"s"},{"type":"text","text":"Hi","citations":[{"type":"web_search_result_location","url":"https://a","title":"A","cited_text":"x"}]},{"type":"tool_use","id":"t","name":"f","input":{"a":1}}],"usage":{"input_tokens":1,"output_tokens":2}}`

type decodedEvent struct {
	ev   *AnthropicStreamEvent
	data []byte
}

func decodeEvents(t *testing.T, events []string) []decodedEvent {
	t.Helper()
	out := make([]decodedEvent, len(events))
	for i, s := range events {
		out[i] = decodedEvent{ev: &AnthropicStreamEvent{}, data: []byte(s)}
		if err := sonic.Unmarshal(out[i].data, out[i].ev); err != nil {
			t.Fatal(err)
		}
	}
	return out
}

func runEvents(state *AnthropicStreamState, evs []decodedEvent) {
	for _, e := range evs {
		state.eventData = e.data
		_, _, _ = e.ev.ToBifrostChatCompletionStream(nil, "", state)
	}
}

func skipAllocUnderRace(t *testing.T) {
	t.Helper()
	if raceEnabled {
		t.Skip("allocation counts are not stable under -race")
	}
}

func TestAnthropicChatStream_NoProviderBlock_AllocBudget(t *testing.T) {
	skipAllocUnderRace(t)
	evs := decodeEvents(t, noProviderBlockStreamEvents)
	state := NewAnthropicStreamState()
	got := testing.AllocsPerRun(200, func() { runEvents(state, evs) })
	t.Logf("stream without provider blocks: %v allocs", got)
	if got > noProviderBlockStreamAllocs {
		t.Fatalf("a stream without provider blocks allocates %v, budget %d", got, noProviderBlockStreamAllocs)
	}
}

func TestAnthropicChatResponse_NoProviderBlock_AllocBudget(t *testing.T) {
	skipAllocUnderRace(t)
	body := []byte(noProviderBlockUnary)
	var resp AnthropicMessageResponse
	if err := sonic.Unmarshal(body, &resp); err != nil {
		t.Fatal(err)
	}
	resp.SetRawBody(body)
	got := testing.AllocsPerRun(200, func() { _ = resp.ToBifrostChatResponse(nil) })
	t.Logf("answer without provider blocks: %v allocs", got)
	if got > noProviderBlockUnaryAllocs {
		t.Fatalf("an answer without provider blocks allocates %v, budget %d", got, noProviderBlockUnaryAllocs)
	}
}

// serverToolUseEvents streams one server_tool_use block whose input arrives in n
// deltas.
func serverToolUseEvents(n int) []string {
	input := `{"query":"` + strings.Repeat("q", 256) + `"}`
	events := []string{startEvent("0", `{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}`)}
	for i := 0; i < n; i++ {
		part := input[i*len(input)/n : (i+1)*len(input)/n]
		quoted, _ := sonic.MarshalString(part)
		events = append(events, `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":`+quoted+`}}`)
	}
	return append(events, stopEvent("0"))
}

// The block costs a constant: the number of deltas never changes the count once
// the stream's scratch has reached the block's size.
func TestAnthropicChatStream_ProviderBlockConstant_AllocBudget(t *testing.T) {
	skipAllocUnderRace(t)
	one := decodeEvents(t, serverToolUseEvents(1))
	many := decodeEvents(t, serverToolUseEvents(256))
	if len(many) != 256+2 {
		t.Fatalf("fixture streams %d events, want 256 deltas", len(many))
	}
	state := NewAnthropicStreamState()
	runEvents(state, many) // warm the scratch to the block's size
	if b, _ := runOnce(state, one); len(b) != 1 {
		t.Fatalf("fixture emitted %d provider blocks", len(b))
	}
	a1 := testing.AllocsPerRun(200, func() { runEvents(state, one) })
	a256 := testing.AllocsPerRun(200, func() { runEvents(state, many) })
	t.Logf("one provider block: %v allocs with 1 delta, %v with 256 deltas", a1, a256)
	if a1 != a256 {
		t.Errorf("1 delta allocates %v, 256 deltas %v: the count depends on the deltas", a1, a256)
	}
	if a1 > providerBlockStreamCeiling {
		t.Errorf("one provider block allocates %v, ceiling %d", a1, providerBlockStreamCeiling)
	}
}

func runOnce(state *AnthropicStreamState, evs []decodedEvent) ([][]byte, error) {
	var blocks [][]byte
	for _, e := range evs {
		state.eventData = e.data
		resp, _, _ := e.ev.ToBifrostChatCompletionStream(nil, "", state)
		if resp != nil {
			for _, pb := range resp.Choices[0].Delta.ProviderBlocks {
				blocks = append(blocks, pb.Block)
			}
		}
	}
	return blocks, nil
}

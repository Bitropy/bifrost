package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"sort"
	"strings"
	"testing"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/tidwall/gjson"
)

func compactJSON(t *testing.T, b []byte) string {
	t.Helper()
	var buf bytes.Buffer
	if err := json.Compact(&buf, b); err != nil {
		t.Fatalf("compact %s: %v", b, err)
	}
	return buf.String()
}

// ContentBlockTypes is the ledger's block domain, so it must be exactly the
// AnthropicContentBlockType constant set of types.go.
func TestContentBlockTypes_MatchesConstBlock(t *testing.T) {
	file, err := parser.ParseFile(token.NewFileSet(), "types.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var declared []string
	for _, decl := range file.Decls {
		gen, ok := decl.(*ast.GenDecl)
		if !ok || gen.Tok != token.CONST {
			continue
		}
		for _, spec := range gen.Specs {
			vs := spec.(*ast.ValueSpec)
			if id, ok := vs.Type.(*ast.Ident); !ok || id.Name != "AnthropicContentBlockType" {
				continue
			}
			for _, v := range vs.Values {
				lit := v.(*ast.BasicLit)
				declared = append(declared, strings.Trim(lit.Value, `"`))
			}
		}
	}
	listed := make([]string, len(ContentBlockTypes))
	for i, ct := range ContentBlockTypes {
		listed[i] = string(ct)
	}
	sort.Strings(declared)
	sort.Strings(listed)
	if strings.Join(declared, ",") != strings.Join(listed, ",") {
		t.Fatalf("ContentBlockTypes drifted from types.go\n declared: %v\n listed:   %v", declared, listed)
	}
}

func TestChatMapsBlock(t *testing.T) {
	mapped := map[AnthropicContentBlockType]bool{
		AnthropicContentBlockTypeText: true, AnthropicContentBlockTypeToolUse: true,
		AnthropicContentBlockTypeThinking: true, AnthropicContentBlockTypeRedactedThinking: true,
	}
	for _, ct := range append(ContentBlockTypes, "future_block", "") {
		if got := ChatMapsBlock(ct); got != mapped[ct] {
			t.Errorf("ChatMapsBlock(%q) = %v, want %v", ct, got, mapped[ct])
		}
	}
}

// unmappedBlockTypes is every block type core does not map to chat, plus one
// type core has never seen.
func unmappedBlockTypes() []AnthropicContentBlockType {
	var out []AnthropicContentBlockType
	for _, ct := range ContentBlockTypes {
		if !ChatMapsBlock(ct) {
			out = append(out, ct)
		}
	}
	return append(out, "future_block")
}

// unmappedBlock carries a member core's struct does not declare, so a re-marshal
// of the typed block would show up as a loss.
func unmappedBlock(ct AnthropicContentBlockType) string {
	return `{"type":"` + string(ct) + `","id":"blk_1","future_member":{"nested":[1,"two",null]}}`
}

func unaryWith(blocks ...string) string {
	return `{"id":"msg_1","type":"message","role":"assistant","model":"claude-x","stop_reason":"end_turn",` +
		`"content":[` + strings.Join(blocks, ",") + `],"usage":{"input_tokens":3,"output_tokens":4}}`
}

func convertUnary(t *testing.T, body string, setRaw bool) *schemas.ChatAssistantMessage {
	t.Helper()
	resp := AcquireAnthropicMessageResponse()
	defer ReleaseAnthropicMessageResponse(resp)
	if err := sonic.Unmarshal([]byte(body), resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if setRaw {
		resp.SetRawBody([]byte(body))
	}
	out := resp.ToBifrostChatResponse(nil)
	return out.Choices[0].Message.ChatAssistantMessage
}

const thinkingBlock = `{"type":"thinking","thinking":"plan","signature":"sig"}`

func TestToBifrostChatResponse_EveryUnmappedTypeBecomesProviderBlock(t *testing.T) {
	for _, ct := range unmappedBlockTypes() {
		t.Run(string(ct), func(t *testing.T) {
			block := unmappedBlock(ct)
			am := convertUnary(t, unaryWith(thinkingBlock, block, `{"type":"text","text":"done"}`), true)
			if am == nil || len(am.ProviderBlocks) != 1 {
				t.Fatalf("want one provider block, got %+v", am)
			}
			pb := am.ProviderBlocks[0]
			if pb.Index != 1 || pb.Provider != schemas.ProviderBlockAnthropic {
				t.Errorf("entry = {index %d, provider %q}, want {1, anthropic}", pb.Index, pb.Provider)
			}
			if got := compactJSON(t, pb.Block); got != block {
				t.Errorf("block = %s, want %s", got, block)
			}
			if len(am.ReasoningDetails) != 1 || am.ReasoningDetails[0].Index != 0 {
				t.Errorf("reasoning_details = %+v, want one at index 0", am.ReasoningDetails)
			}
		})
	}
}

// The shared counter runs over reasoning details and provider blocks in content
// order; the spec's example answer.
func TestToBifrostChatResponse_SharedIndex(t *testing.T) {
	stu := `{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"q"}}`
	wsr := `{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[{"type":"web_search_result","url":"https://a","title":"A","encrypted_content":"E","page_age":null}]}`
	am := convertUnary(t, unaryWith(thinkingBlock, stu, wsr, `{"type":"text","text":"answer"}`), true)
	if len(am.ReasoningDetails) != 1 || am.ReasoningDetails[0].Index != 0 {
		t.Fatalf("reasoning_details = %+v", am.ReasoningDetails)
	}
	if len(am.ProviderBlocks) != 2 || am.ProviderBlocks[0].Index != 1 || am.ProviderBlocks[1].Index != 2 {
		t.Fatalf("provider_blocks = %+v", am.ProviderBlocks)
	}
	if compactJSON(t, am.ProviderBlocks[0].Block) != stu || compactJSON(t, am.ProviderBlocks[1].Block) != wsr {
		t.Errorf("blocks = %s / %s", am.ProviderBlocks[0].Block, am.ProviderBlocks[1].Block)
	}
	if len(am.ToolCalls) != 0 {
		t.Errorf("server_tool_use leaked as a tool call: %+v", am.ToolCalls)
	}
}

// Without the raw body the converter has nothing to copy: it emits no provider
// block, and the answer is what it was before provider blocks existed.
func TestToBifrostChatResponse_NoRawBodyNoProviderBlock(t *testing.T) {
	am := convertUnary(t, unaryWith(thinkingBlock, unmappedBlock("server_tool_use"), `{"type":"text","text":"x"}`), false)
	if am == nil || len(am.ProviderBlocks) != 0 {
		t.Fatalf("provider blocks without a raw body: %+v", am)
	}
}

// A chat stream fed event by event as the stream loop does: decoded with the
// production decoder, with the event's data bytes set on the state.
type chatStreamOut struct {
	chunks  []*schemas.BifrostChatResponse
	atEvent []int
	state   *AnthropicStreamState
}

func feedChatStream(t *testing.T, events []string) chatStreamOut {
	t.Helper()
	out := chatStreamOut{state: NewAnthropicStreamState()}
	for i, raw := range events {
		data := []byte(raw)
		var ev AnthropicStreamEvent
		if err := decodeAnthropicStreamEvent(data, &ev); err != nil {
			t.Fatalf("event %d: %v", i, err)
		}
		out.state.eventData = data
		resp, bErr, _ := ev.ToBifrostChatCompletionStream(nil, "", out.state)
		if bErr != nil {
			t.Fatalf("event %d: %+v", i, bErr)
		}
		if resp != nil {
			out.chunks = append(out.chunks, resp)
			out.atEvent = append(out.atEvent, i)
		}
	}
	return out
}

// providerBlockChunks returns the chunks carrying provider blocks and the event
// index each was emitted on.
func (o chatStreamOut) providerBlockChunks() ([]schemas.ChatProviderBlock, []int) {
	var blocks []schemas.ChatProviderBlock
	var at []int
	for i, c := range o.chunks {
		if pbs := c.Choices[0].ChatStreamResponseChoice.Delta.ProviderBlocks; len(pbs) > 0 {
			blocks = append(blocks, pbs...)
			at = append(at, o.atEvent[i])
		}
	}
	return blocks, at
}

func startEvent(index, block string) string {
	return `{"type":"content_block_start","index":` + index + `,"content_block":` + block + `}`
}

func stopEvent(index string) string {
	return `{"type":"content_block_stop","index":` + index + `}`
}

func TestChatStream_EveryUnmappedTypeIsOneDeltaAtStop(t *testing.T) {
	for _, ct := range unmappedBlockTypes() {
		t.Run(string(ct), func(t *testing.T) {
			block := unmappedBlock(ct)
			out := feedChatStream(t, []string{
				startEvent("0", `{"type":"thinking","thinking":""}`),
				`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"plan"}}`,
				stopEvent("0"),
				startEvent("1", block),
				stopEvent("1"),
				startEvent("2", `{"type":"text","text":""}`),
				`{"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"done"}}`,
				stopEvent("2"),
			})
			blocks, at := out.providerBlockChunks()
			if len(blocks) != 1 || len(at) != 1 || at[0] != 4 {
				t.Fatalf("want exactly one provider block on event 4 (its stop), got %+v at %v", blocks, at)
			}
			if blocks[0].Index != 1 || blocks[0].Provider != schemas.ProviderBlockAnthropic {
				t.Errorf("entry = {index %d, provider %q}", blocks[0].Index, blocks[0].Provider)
			}
			if got := compactJSON(t, blocks[0].Block); got != block {
				t.Errorf("block = %s, want %s", got, block)
			}
			if out.state.providerBlockDrops != (schemas.ProviderBlockDrops{}) {
				t.Errorf("drops = %+v, want none", out.state.providerBlockDrops)
			}
		})
	}
}

// The streamed members fold into the start block, so the stream's block equals
// the unary block of the same answer.
func TestChatStream_FoldsStreamedMembers(t *testing.T) {
	tests := []struct {
		name   string
		events []string
		unary  string
	}{
		{"server_tool_use input", []string{
			startEvent("0", `{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}`),
			`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":""}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query"}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\": \"bbc \\\"top\\\"\"}"}}`,
			stopEvent("0"),
		}, `{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"bbc \"top\""}}`},
		{"mcp_tool_use input", []string{
			startEvent("0", `{"type":"mcp_tool_use","id":"mcptoolu_1","name":"lookup","server_name":"m","input":{}}`),
			`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"id\":7}"}}`,
			stopEvent("0"),
		}, `{"type":"mcp_tool_use","id":"mcptoolu_1","name":"lookup","server_name":"m","input":{"id":7}}`},
		{"server_tool_use without input member", []string{
			startEvent("0", `{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search"}`),
			`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"q\":1}"}}`,
			stopEvent("0"),
		}, `{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"q":1}}`},
		{"no delta keeps the start block", []string{
			startEvent("0", `{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"q":2}}`),
			`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":""}}`,
			stopEvent("0"),
		}, `{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"q":2}}`},
		{"compaction content", []string{
			startEvent("0", `{"type":"compaction","content":""}`),
			`{"type":"content_block_delta","index":0,"delta":{"type":"compaction_delta","content":"Sum"}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"compaction_delta","content":"mary \"q\" é"}}`,
			stopEvent("0"),
		}, `{"type":"compaction","content":"Summary \"q\" é"}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := feedChatStream(t, tt.events)
			blocks, at := out.providerBlockChunks()
			if len(blocks) != 1 || at[0] != len(tt.events)-1 {
				t.Fatalf("want one provider block at the stop, got %+v at %v", blocks, at)
			}
			var want, got any
			if err := json.Unmarshal([]byte(tt.unary), &want); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(blocks[0].Block, &got); err != nil {
				t.Fatalf("assembled block is not JSON: %s: %v", blocks[0].Block, err)
			}
			if gb, _ := json.Marshal(got); string(gb) != mustMarshal(t, want) {
				t.Errorf("assembled %s, want %s", blocks[0].Block, tt.unary)
			}
			if len(out.chunks) != 1 {
				t.Errorf("deltas of an unmapped block emitted %d chunks, want only the stop chunk", len(out.chunks))
			}
		})
	}
}

func mustMarshal(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// A delta type the converter does not fold is dropped and counted; the block is
// still emitted as assembled.
func TestChatStream_UnknownDeltaDroppedAndCounted(t *testing.T) {
	out := feedChatStream(t, []string{
		startEvent("0", `{"type":"server_tool_use","id":"s","name":"web_search","input":{}}`),
		`{"type":"content_block_delta","index":0,"delta":{"type":"future_delta","x":"y"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"q\":1}"}}`,
		stopEvent("0"),
	})
	blocks, _ := out.providerBlockChunks()
	if len(blocks) != 1 || compactJSON(t, blocks[0].Block) != `{"type":"server_tool_use","id":"s","name":"web_search","input":{"q":1}}` {
		t.Fatalf("blocks = %+v", blocks)
	}
	if out.state.providerBlockDrops != (schemas.ProviderBlockDrops{UnknownDelta: 1}) {
		t.Errorf("drops = %+v, want one unknown delta", out.state.providerBlockDrops)
	}
}

// Anthropic never interleaves blocks; a start while an unmapped block is open
// drops the open block and counts it, and the new block streams normally.
func TestChatStream_InterleavedStartDropsOpenBlock(t *testing.T) {
	out := feedChatStream(t, []string{
		startEvent("0", `{"type":"server_tool_use","id":"s","name":"web_search","input":{}}`),
		startEvent("1", `{"type":"web_search_tool_result","tool_use_id":"s","content":[]}`),
		stopEvent("1"),
		stopEvent("0"),
	})
	blocks, at := out.providerBlockChunks()
	if len(blocks) != 1 || at[0] != 2 || compactJSON(t, blocks[0].Block) != `{"type":"web_search_tool_result","tool_use_id":"s","content":[]}` {
		t.Fatalf("blocks = %+v at %v", blocks, at)
	}
	if blocks[0].Index != 1 {
		t.Errorf("index = %d, want 1 (the dropped block keeps its slot)", blocks[0].Index)
	}
	if out.state.providerBlockDrops != (schemas.ProviderBlockDrops{Interleaved: 1}) {
		t.Errorf("drops = %+v, want one interleaved", out.state.providerBlockDrops)
	}
}

// The unary and the streamed answer of the same turn give equal blocks with the
// same indices.
func TestChatStream_MatchesUnary(t *testing.T) {
	stu := `{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"q"}}`
	wsr := `{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[]}`
	am := convertUnary(t, unaryWith(thinkingBlock, stu, wsr, `{"type":"text","text":"a"}`), true)
	out := feedChatStream(t, []string{
		startEvent("0", `{"type":"thinking","thinking":""}`),
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"plan"}}`,
		stopEvent("0"),
		startEvent("1", `{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}`),
		`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"q\"}"}}`,
		stopEvent("1"),
		startEvent("2", wsr),
		stopEvent("2"),
	})
	blocks, _ := out.providerBlockChunks()
	if len(blocks) != len(am.ProviderBlocks) {
		t.Fatalf("stream %d blocks, unary %d", len(blocks), len(am.ProviderBlocks))
	}
	for i := range blocks {
		if blocks[i].Index != am.ProviderBlocks[i].Index || compactJSON(t, blocks[i].Block) != compactJSON(t, am.ProviderBlocks[i].Block) {
			t.Errorf("block %d: stream {%d %s}, unary {%d %s}", i, blocks[i].Index, blocks[i].Block, am.ProviderBlocks[i].Index, am.ProviderBlocks[i].Block)
		}
	}
}

// The real stream loop hands the converter the event bytes and puts the drop
// counts on the final chunk.
func TestChatCompletionStream_ProviderBlocksAndDropsOnFinalChunk(t *testing.T) {
	sse := func(data string) string { return "event: x\ndata: " + data + "\n\n" }
	body := sse(`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-x","content":[],"usage":{"input_tokens":5,"output_tokens":0}}}`) +
		sse(startEvent("0", `{"type":"server_tool_use","id":"s","name":"web_search","input":{}}`)) +
		sse(`{"type":"content_block_delta","index":0,"delta":{"type":"future_delta","x":1}}`) +
		sse(`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"q\":1}"}}`) +
		sse(stopEvent("0")) +
		sse(`{"type":"message_delta","delta":{"stop_reason":"pause_turn"},"usage":{"output_tokens":3,"server_tool_use":{"web_search_requests":1,"web_fetch_requests":2}}}`) +
		sse(`{"type":"message_stop"}`)
	server := anthropicSSEServer(t, body, false)
	t.Cleanup(server.Close)
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	stream, bErr := newTruncationTestProvider(server.URL).ChatCompletionStream(ctx, truncationPassthroughPostHook, nil,
		schemas.Key{Value: *schemas.NewSecretVar("k")}, truncationChatRequest())
	if bErr != nil {
		t.Fatal(bErr)
	}
	chunks := collectTruncationChunks(t, stream)
	var blocks []schemas.ChatProviderBlock
	for _, c := range chunks {
		if c.BifrostError != nil {
			t.Fatalf("stream error: %+v", c.BifrostError)
		}
		if r := c.BifrostChatResponse; r != nil && len(r.Choices) > 0 && r.Choices[0].ChatStreamResponseChoice != nil && r.Choices[0].Delta != nil {
			blocks = append(blocks, r.Choices[0].Delta.ProviderBlocks...)
		}
	}
	if len(blocks) != 1 || compactJSON(t, blocks[0].Block) != `{"type":"server_tool_use","id":"s","name":"web_search","input":{"q":1}}` {
		t.Fatalf("blocks = %+v", blocks)
	}
	final := chunks[len(chunks)-1].BifrostChatResponse
	if final.ExtraFields.ProviderBlockDrops != (schemas.ProviderBlockDrops{UnknownDelta: 1}) {
		t.Errorf("final chunk drops = %+v", final.ExtraFields.ProviderBlockDrops)
	}
	if final.Usage == nil || final.Usage.ToolUsage == nil || final.Usage.ToolUsage.WebFetch == nil || final.Usage.ToolUsage.WebFetch.NumRequests != 2 ||
		final.Usage.ToolUsage.WebSearch == nil || final.Usage.ToolUsage.WebSearch.NumRequests != 1 {
		t.Errorf("final usage tool_usage = %+v", final.Usage)
	}
	if out, _ := sonic.Marshal(final); bytes.Contains(out, []byte("provider_block_drops")) || bytes.Contains(out, []byte("ProviderBlockDrops")) {
		t.Errorf("drop counts serialized: %s", out)
	}
}

func TestToBifrostChatResponse_WebFetchUsage(t *testing.T) {
	body := `{"id":"m","type":"message","role":"assistant","model":"c","content":[{"type":"text","text":"x"}],` +
		`"usage":{"input_tokens":1,"output_tokens":1,"server_tool_use":{"web_search_requests":0,"web_fetch_requests":3}}}`
	var resp AnthropicMessageResponse
	if err := sonic.Unmarshal([]byte(body), &resp); err != nil {
		t.Fatal(err)
	}
	u := resp.ToBifrostChatResponse(nil).Usage
	if u.ToolUsage == nil || u.ToolUsage.WebFetch == nil || u.ToolUsage.WebFetch.NumRequests != 3 || u.ToolUsage.WebSearch != nil {
		t.Errorf("tool_usage = %+v", u.ToolUsage)
	}
}

// replayRequest is an assistant turn carrying reasoning details and provider
// blocks, as a chat client sends a Claude answer back.
func replayRequest(provider schemas.ModelProvider, details []schemas.ChatReasoningDetails, blocks []schemas.ChatProviderBlock) *schemas.BifrostChatRequest {
	return &schemas.BifrostChatRequest{
		Provider: provider,
		Model:    "claude-sonnet-4-5",
		Input: []schemas.ChatMessage{
			{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("find it")}},
			{
				Role:    schemas.ChatMessageRoleAssistant,
				Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("found")},
				ChatAssistantMessage: &schemas.ChatAssistantMessage{
					ReasoningDetails: details,
					ProviderBlocks:   blocks,
					ToolCalls: []schemas.ChatAssistantMessageToolCall{{
						ID: schemas.Ptr("toolu_1"), Type: schemas.Ptr("function"),
						Function: schemas.ChatAssistantMessageToolCallFunction{Name: schemas.Ptr("f"), Arguments: `{"a":1}`},
					}},
				},
			},
			{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("go on")}},
		},
	}
}

const (
	replaySTU = `{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"tok__abc"},"future_member":[1]}`
	replayWSR = `{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[]}`
)

func pb(index int, block string) schemas.ChatProviderBlock {
	return schemas.ChatProviderBlock{Index: index, Provider: schemas.ProviderBlockAnthropic, Block: json.RawMessage(block)}
}

func thinkingDetail(index int, text string) schemas.ChatReasoningDetails {
	return schemas.ChatReasoningDetails{Index: index, Type: schemas.BifrostReasoningDetailsTypeText, Text: schemas.Ptr(text), Signature: schemas.Ptr("sig")}
}

// assistantContent returns the wire content of the converted assistant turn.
func assistantContent(t *testing.T, req *schemas.BifrostChatRequest) []json.RawMessage {
	t.Helper()
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	out, err := ToAnthropicChatRequest(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	wire, err := out.Messages[1].Content.MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}
	var blocks []json.RawMessage
	if err := json.Unmarshal(wire, &blocks); err != nil {
		t.Fatalf("content %s: %v", wire, err)
	}
	return blocks
}

func blockTypes(blocks []json.RawMessage) []string {
	out := make([]string, len(blocks))
	for i, b := range blocks {
		out[i] = gjson.GetBytes(b, "type").String() + ":" + gjson.GetBytes(b, "thinking").String()
	}
	return out
}

func TestToAnthropicChatRequest_ReinsertsProviderBlocksByIndex(t *testing.T) {
	tests := []struct {
		name    string
		details []schemas.ChatReasoningDetails
		blocks  []schemas.ChatProviderBlock
		want    []string
	}{
		{"leading run before text and tool_use",
			[]schemas.ChatReasoningDetails{thinkingDetail(0, "a")}, []schemas.ChatProviderBlock{pb(1, replaySTU), pb(2, replayWSR)},
			[]string{"thinking:a", "server_tool_use:", "web_search_tool_result:", "text:", "tool_use:"}},
		{"merged by index",
			[]schemas.ChatReasoningDetails{thinkingDetail(1, "a"), thinkingDetail(3, "b")}, []schemas.ChatProviderBlock{pb(0, replaySTU), pb(2, replayWSR)},
			[]string{"server_tool_use:", "thinking:a", "web_search_tool_result:", "thinking:b", "text:", "tool_use:"}},
		{"tie puts reasoning first",
			[]schemas.ChatReasoningDetails{thinkingDetail(1, "a")}, []schemas.ChatProviderBlock{pb(1, replaySTU), pb(0, replayWSR)},
			[]string{"web_search_tool_result:", "thinking:a", "server_tool_use:", "text:", "tool_use:"}},
		{"provider blocks only",
			nil, []schemas.ChatProviderBlock{pb(0, replaySTU)},
			[]string{"server_tool_use:", "text:", "tool_use:"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			blocks := assistantContent(t, replayRequest(schemas.Anthropic, tt.details, tt.blocks))
			if got := blockTypes(blocks); strings.Join(got, ",") != strings.Join(tt.want, ",") {
				t.Fatalf("order = %v, want %v", got, tt.want)
			}
			for _, b := range blocks {
				switch gjson.GetBytes(b, "type").String() {
				case "server_tool_use":
					if compactJSON(t, b) != replaySTU {
						t.Errorf("replayed block = %s, want %s byte for byte", b, replaySTU)
					}
				case "web_search_tool_result":
					if compactJSON(t, b) != replayWSR {
						t.Errorf("replayed block = %s, want %s", b, replayWSR)
					}
				}
			}
		})
	}
}

// Malformed entries fail the conversion for embedders other than the router,
// naming the entry's position and echoing nothing the client wrote.
func TestToAnthropicChatRequest_RejectsMalformedProviderBlocks(t *testing.T) {
	for name, entry := range map[string]schemas.ChatProviderBlock{
		"foreign provider": {Index: 0, Provider: "zzclientprov", Block: json.RawMessage(replaySTU)},
		"non-object block": {Index: 0, Provider: schemas.ProviderBlockAnthropic, Block: json.RawMessage(`"zzclienttext"`)},
	} {
		t.Run(name, func(t *testing.T) {
			ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
			_, err := ToAnthropicChatRequest(ctx, replayRequest(schemas.Anthropic, nil, []schemas.ChatProviderBlock{pb(0, replayWSR), entry}))
			if err == nil {
				t.Fatal("want a request error")
			}
			if !strings.Contains(err.Error(), "messages[1].provider_blocks[1]") {
				t.Errorf("error %q does not name the entry", err)
			}
			if strings.Contains(err.Error(), "zzclient") {
				t.Errorf("error %q echoes client text", err)
			}
		})
	}
}

// A replayed block's cache_control is the typed field: the Vertex scope strip
// treats it exactly as it treats a typed block, and the marshal re-encodes it
// as the block's last member.
func TestReplayedBlockCacheControl(t *testing.T) {
	const scoped = `{"type":"server_tool_use","id":"s","cache_control":{"type":"ephemeral","scope":"user"},"name":"web_search","input":{}}`
	typedReq := func(provider schemas.ModelProvider) *schemas.BifrostChatRequest {
		req := replayRequest(provider, nil, nil)
		req.Input[1].Content = &schemas.ChatMessageContent{ContentBlocks: []schemas.ChatContentBlock{{
			Type: schemas.ChatContentBlockTypeText, Text: schemas.Ptr("found"),
			CacheControl: &schemas.CacheControl{Type: schemas.CacheControlTypeEphemeral, Scope: schemas.Ptr("user")},
		}}}
		return req
	}
	build := func(req *schemas.BifrostChatRequest) ([]byte, Feature) {
		ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
		body, bErr := BuildAnthropicChatRequestBody(ctx, req, AnthropicRequestBuildConfig{Provider: req.Provider})
		if bErr != nil {
			t.Fatalf("build: %+v", bErr.Error)
		}
		return body, RemovedFeatures(ctx)
	}
	for _, provider := range []schemas.ModelProvider{schemas.Vertex, schemas.Anthropic} {
		t.Run(string(provider), func(t *testing.T) {
			typedBody, typedRemoved := build(typedReq(provider))
			replayBody, replayRemoved := build(replayRequest(provider, nil, []schemas.ChatProviderBlock{pb(0, scoped)}))
			typedCC := gjson.GetBytes(typedBody, "messages.1.content.0.cache_control").Raw
			replayCC := gjson.GetBytes(replayBody, "messages.1.content.0.cache_control").Raw
			if typedCC == "" || typedCC != replayCC {
				t.Errorf("cache_control typed %s, replayed %s", typedCC, replayCC)
			}
			if typedRemoved&FeatureCacheControlScope != replayRemoved&FeatureCacheControlScope {
				t.Errorf("scope report typed %v, replayed %v", typedRemoved, replayRemoved)
			}
			if provider == schemas.Vertex && replayRemoved&FeatureCacheControlScope == 0 {
				t.Error("Vertex did not report the replayed block's scope strip")
			}
			block := gjson.GetBytes(replayBody, "messages.1.content.0").Raw
			if !strings.HasSuffix(block, `"cache_control":`+replayCC+`}`) {
				t.Errorf("cache_control is not the replayed block's last member: %s", block)
			}
		})
	}

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	out, err := ToAnthropicChatRequest(ctx, replayRequest(schemas.Anthropic, nil, []schemas.ChatProviderBlock{pb(0, scoped), pb(1, replayWSR)}))
	if err != nil {
		t.Fatal(err)
	}
	moved := out.Messages[1].Content.ContentBlocks[0]
	if bytes.Contains(moved.Raw, []byte("cache_control")) || moved.CacheControl == nil || moved.Type != AnthropicContentBlockTypeServerToolUse {
		t.Errorf("re-insertion kept cache_control in Raw or lost the typed fields: %s %+v", moved.Raw, moved)
	}
	plain := out.Messages[1].Content.ContentBlocks[1]
	if string(plain.Raw) != replayWSR {
		t.Errorf("a block without cache_control is not kept byte for byte: %s", plain.Raw)
	}
}

// With no replayed block the content marshal is today's.
func TestAnthropicContentMarshal_NoRawBlockUnchanged(t *testing.T) {
	content := AnthropicContent{ContentBlocks: []AnthropicContentBlock{
		{Type: AnthropicContentBlockTypeThinking, Thinking: schemas.Ptr("t"), Signature: schemas.Ptr("s")},
		{Type: AnthropicContentBlockTypeText, Text: schemas.Ptr("x"), CacheControl: &schemas.CacheControl{Type: schemas.CacheControlTypeEphemeral}},
		{Type: AnthropicContentBlockTypeToolUse, ID: schemas.Ptr("t1"), Name: schemas.Ptr("f"), Input: json.RawMessage(`{"a":1}`)},
	}}
	got, err := content.MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}
	want, err := sonic.Marshal(content.ContentBlocks)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Errorf("got  %s\nwant %s", got, want)
	}
}

// Vertex and Azure Claude build through the same converter, so the replayed
// turn reaches every Claude route identically.
func TestBuildAnthropicChatRequestBody_ReplayIdenticalOnClaudeRoutes(t *testing.T) {
	var want string
	for _, provider := range []schemas.ModelProvider{schemas.Anthropic, schemas.Azure, schemas.Vertex} {
		req := replayRequest(provider, []schemas.ChatReasoningDetails{thinkingDetail(0, "a")}, []schemas.ChatProviderBlock{pb(1, replaySTU), pb(2, replayWSR)})
		ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
		body, bErr := BuildAnthropicChatRequestBody(ctx, req, AnthropicRequestBuildConfig{Provider: provider})
		if bErr != nil {
			t.Fatalf("%s: %+v", provider, bErr.Error)
		}
		got := gjson.GetBytes(body, "messages.1").Raw
		if !strings.Contains(got, replaySTU) {
			t.Errorf("%s: replayed block missing: %s", provider, got)
		}
		if want == "" {
			want = got
		} else if got != want {
			t.Errorf("%s assistant turn differs from native\n got  %s\n want %s", provider, got, want)
		}
	}
}

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
	// A fresh state per run, as each stream has: the state and its maps stay off the heap
	// only while nothing in the converter makes them escape.
	got := testing.AllocsPerRun(200, func() { runEvents(NewAnthropicStreamState(), evs) })
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

const (
	citeChar    = `{"type":"char_location","cited_text":"alpha","document_index":0,"document_title":"Doc","start_char_index":0,"end_char_index":5}`
	citePage    = `{"type":"page_location","cited_text":"beta","document_index":1,"start_page_number":2,"end_page_number":3}`
	citeFuture  = `{"type":"future_location","cited_text":"gamma","future_member":{"x":[1]}}`
	citeWeb     = `{"type":"web_search_result_location","url":"https://a.example/x","title":"A","encrypted_index":"E","cited_text":"web"}`
	citedFirst  = "Alpha and beta."
	citedSecond = " Web says so."
)

// wantProviderCitation is one expected provider_citation annotation.
type wantProviderCitation struct {
	start, end int
	citation   string
}

func checkProviderCitations(t *testing.T, anns []schemas.ChatAssistantMessageAnnotation, wantURL int, want []wantProviderCitation) {
	t.Helper()
	// Content order; within a block url_citations come first.
	wantOrder := []string{"provider_citation", "provider_citation", "url_citation", "provider_citation"}
	var urls, providers []schemas.ChatAssistantMessageAnnotation
	var order []string
	for _, a := range anns {
		order = append(order, a.Type)
		switch a.Type {
		case "url_citation":
			urls = append(urls, a)
		case schemas.ChatAnnotationTypeProviderCitation:
			providers = append(providers, a)
		}
	}
	if fmt.Sprint(order) != fmt.Sprint(wantOrder) {
		t.Errorf("annotation order = %v, want %v", order, wantOrder)
	}
	if len(urls) != wantURL {
		t.Errorf("url_citation annotations = %d, want %d", len(urls), wantURL)
	}
	if len(providers) != len(want) {
		t.Fatalf("provider_citation annotations = %+v, want %d", providers, len(want))
	}
	for i, w := range want {
		a := providers[i]
		if a.Provider != schemas.ProviderBlockAnthropic || a.StartIndex == nil || a.EndIndex == nil || *a.StartIndex != w.start || *a.EndIndex != w.end {
			t.Errorf("annotation %d = {provider %q, span %v..%v}, want {anthropic, %d..%d}", i, a.Provider, a.StartIndex, a.EndIndex, w.start, w.end)
		}
		if got := compactJSON(t, a.Citation); got != w.citation {
			t.Errorf("annotation %d citation = %s, want %s", i, got, w.citation)
		}
	}
}

var wantDocCitations = []wantProviderCitation{
	{0, len(citedFirst), citeChar},
	{0, len(citedFirst), citePage},
	{len(citedFirst), len(citedFirst) + len(citedSecond), citeFuture},
}

func TestToBifrostChatResponse_ProviderCitations(t *testing.T) {
	body := unaryWith(
		`{"type":"text","text":"`+citedFirst+`","citations":[`+citeChar+`,`+citePage+`]}`,
		`{"type":"text","text":"`+citedSecond+`","citations":[`+citeWeb+`,`+citeFuture+`]}`,
	)
	am := convertUnary(t, body, true)
	if am == nil {
		t.Fatal("no assistant message")
	}
	checkProviderCitations(t, am.Annotations, 1, wantDocCitations)
}

func TestChatStream_ProviderCitationsAtBlockStop(t *testing.T) {
	out := feedChatStream(t, []string{
		startEvent("0", `{"type":"text","text":"","citations":[`+citeChar+`]}`),
		`{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":` + citePage + `}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"` + citedFirst + `"}}`,
		stopEvent("0"),
		startEvent("1", `{"type":"text","text":""}`),
		`{"type":"content_block_delta","index":1,"delta":{"type":"citations_delta","citation":` + citeWeb + `}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"citations_delta","citation":` + citeFuture + `}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"` + citedSecond + `"}}`,
		stopEvent("1"),
	})
	var anns []schemas.ChatAssistantMessageAnnotation
	for i, c := range out.chunks {
		if a := c.Choices[0].Delta.Annotations; len(a) > 0 {
			if ev := out.atEvent[i]; ev != 3 && ev != 8 {
				t.Errorf("annotations emitted on event %d, want only the blocks' stops", ev)
			}
			anns = append(anns, a...)
		}
	}
	checkProviderCitations(t, anns, 1, wantDocCitations)
}

func TestProviderCitationEncoding(t *testing.T) {
	start, end := 0, 5
	got, err := sonic.Marshal(schemas.ChatAssistantMessageAnnotation{
		Type: schemas.ChatAnnotationTypeProviderCitation, Provider: schemas.ProviderBlockAnthropic,
		StartIndex: &start, EndIndex: &end, Citation: []byte(citeChar),
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := `{"type":"provider_citation","provider":"anthropic","start_index":0,"end_index":5,"citation":` + citeChar + `}`; string(got) != want {
		t.Errorf("provider_citation = %s\nwant %s", got, want)
	}
	// url_citation bytes are what they were before the provider members existed.
	got, err = sonic.Marshal(schemas.ChatAssistantMessageAnnotation{
		Type:        "url_citation",
		URLCitation: schemas.ChatAssistantMessageAnnotationCitation{StartIndex: 1, EndIndex: 4, Title: "A", URL: schemas.Ptr("https://a"), Text: schemas.Ptr("x")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := `{"type":"url_citation","url_citation":{"start_index":1,"end_index":4,"title":"A","url":"https://a","text":"x"}}`; string(got) != want {
		t.Errorf("url_citation = %s\nwant %s", got, want)
	}
}

// An answer whose only annotations are url_citations encodes with the base pin's
// allocations.
func TestChatAnnotationEncode_URLCitationOnly_AllocBudget(t *testing.T) {
	skipAllocUnderRace(t)
	const urlCitationEncodeAllocs = 19
	var resp AnthropicMessageResponse
	if err := sonic.Unmarshal([]byte(noProviderBlockUnary), &resp); err != nil {
		t.Fatal(err)
	}
	out := resp.ToBifrostChatResponse(nil)
	got := testing.AllocsPerRun(200, func() { _, _ = sonic.Marshal(out) })
	t.Logf("url_citation-only answer encodes in %v allocs", got)
	if got > urlCitationEncodeAllocs {
		t.Fatalf("encoding allocates %v, budget %d", got, urlCitationEncodeAllocs)
	}
}

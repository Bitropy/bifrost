package anthropic

import (
	"bytes"
	"context"
	"fmt"
	"maps"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/core/internal/providerblocktest"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
	"github.com/valyala/fasthttp"
)

// responsesUnaryFrom serves body from an httptest upstream and converts it through
// HandleAnthropicResponsesRequest, the path every Anthropic-dialect provider shares.
func responsesUnaryFrom(t *testing.T, body string) (*schemas.BifrostResponsesResponse, *schemas.BifrostContext) {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(server.Close)
	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	request := &schemas.BifrostResponsesRequest{Provider: schemas.Anthropic, Model: "claude-sonnet-4-6", Input: makeSimpleInput("hi")}
	resp, bifrostErr := HandleAnthropicResponsesRequest(ctx, &fasthttp.Client{}, server.URL, request,
		AnthropicRequestBuildConfig{Provider: schemas.Anthropic, Model: "claude-sonnet-4-6"},
		map[string]string{}, nil, nil, truncationTestLogger{})
	if bifrostErr != nil {
		t.Fatalf("unary: %s", bifrostErr.Error.Message)
	}
	return resp, ctx
}

// responsesStreamFrom serves frames as an SSE stream and collects the converted events.
func responsesStreamFrom(t *testing.T, frames []string) ([]*schemas.BifrostResponsesStreamResponse, *schemas.BifrostResponsesStreamResponse) {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, frame := range frames {
			_, _ = fmt.Fprintf(w, "event: %s\ndata: %s\n\n", gjson.Get(frame, "type").String(), frame)
		}
	}))
	t.Cleanup(server.Close)
	ctx, cancel := schemas.NewBifrostContextWithCancel(context.Background())
	t.Cleanup(cancel)
	body, buildErr := BuildAnthropicResponsesRequestBody(ctx, &schemas.BifrostResponsesRequest{
		Provider: schemas.Anthropic, Model: "claude-sonnet-4-6", Input: makeSimpleInput("hi"),
	}, AnthropicRequestBuildConfig{Provider: schemas.Anthropic, IsStreaming: true})
	if buildErr != nil {
		t.Fatal(buildErr)
	}
	stream, err := HandleAnthropicResponsesStream(ctx, &fasthttp.Client{}, server.URL, body, nil, nil, 30, nil, false, false, schemas.Anthropic, truncationPassthroughPostHook, nil, nil, truncationTestLogger{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	var events []*schemas.BifrostResponsesStreamResponse
	for _, chunk := range collectTruncationChunks(t, stream) {
		if chunk.BifrostError != nil {
			t.Fatalf("stream error: %+v", chunk.BifrostError.Error)
		}
		if chunk.BifrostResponsesStreamResponse != nil {
			events = append(events, chunk.BifrostResponsesStreamResponse)
		}
	}
	if len(events) == 0 {
		t.Fatal("stream produced no events")
	}
	return events, events[len(events)-1]
}

func streamFramesWith(usage string, blocks ...string) []string {
	frames := []string{`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[],"usage":{"input_tokens":3,"output_tokens":1}}}`}
	for i, b := range blocks {
		idx := fmt.Sprint(i)
		frames = append(frames, `{"type":"content_block_start","index":`+idx+`,"content_block":`+b+`}`, `{"type":"content_block_stop","index":`+idx+`}`)
	}
	if usage == "" {
		usage = `{"output_tokens":4}`
	}
	return append(frames,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":`+usage+`}`,
		`{"type":"message_stop"}`)
}

func marshalResponse(t *testing.T, r *schemas.BifrostResponsesResponse) string {
	t.Helper()
	b, err := sonic.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestResponsesUsage_WebFetch(t *testing.T) {
	cases := []struct {
		name, serverToolUse string
		wantToolUsage       string
		wantSearchQueries   int64
	}{
		{"search and fetch", `{"web_search_requests":2,"web_fetch_requests":3}`, `{"web_search":{"num_requests":2},"web_fetch":{"num_requests":3}}`, 2},
		{"fetch only", `{"web_search_requests":0,"web_fetch_requests":3}`, `{"web_fetch":{"num_requests":3}}`, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			usage := `{"input_tokens":3,"output_tokens":4,"server_tool_use":` + tc.serverToolUse +
				`,"iterations":[{"type":"message","input_tokens":3,"output_tokens":4,"server_tool_use":` + tc.serverToolUse + `}]}`
			check := func(mode, out string) {
				t.Helper()
				if got := gjson.Get(out, "tool_usage").Raw; got != tc.wantToolUsage {
					t.Errorf("%s: tool_usage = %s, want %s", mode, got, tc.wantToolUsage)
				}
				q := gjson.Get(out, "usage.output_tokens_details.num_search_queries")
				if tc.wantSearchQueries == 0 && q.Exists() {
					t.Errorf("%s: num_search_queries = %s, want absent", mode, q.Raw)
				} else if tc.wantSearchQueries > 0 && q.Int() != tc.wantSearchQueries {
					t.Errorf("%s: num_search_queries = %s, want %d", mode, q.Raw, tc.wantSearchQueries)
				}
			}

			unary, _ := responsesUnaryFrom(t, `{"id":"msg_1","type":"message","role":"assistant","model":"claude-x","stop_reason":"end_turn","content":[{"type":"text","text":"hi"}],"usage":`+usage+`}`)
			out := marshalResponse(t, unary)
			check("unary", out)
			if got := gjson.Get(out, "usage.iterations.0.tool_usage.web_fetch.num_requests").Int(); got != 3 {
				t.Errorf("unary iteration web_fetch = %d, want 3 (%s)", got, gjson.Get(out, "usage.iterations.0").Raw)
			}

			_, last := responsesStreamFrom(t, streamFramesWith(usage, `{"type":"text","text":"hi"}`))
			if last.Response == nil {
				t.Fatal("stream: terminal event has no response")
			}
			check("stream", marshalResponse(t, last.Response))

			var acc schemas.ResponsesResponseUsage
			var billed schemas.BifrostLLMUsage
			var au AnthropicUsage
			if err := sonic.UnmarshalString(usage, &au); err != nil {
				t.Fatal(err)
			}
			accumulateAnthropicResponsesUsage(&acc, &billed, &au, anthropicPromptCounts{})
			if billed.ToolUsage == nil || billed.ToolUsage.WebFetch == nil || billed.ToolUsage.WebFetch.NumRequests != 3 {
				t.Errorf("billed web_fetch = %+v, want 3", billed.ToolUsage)
			}
		})
	}
}

// responsesBlockFixture is a realistic top-level answer block of one type, after the call
// blocks its arm needs to place it.
type responsesBlockFixture struct{ pre, block string }

func responsesBlockFixtures() map[AnthropicContentBlockType]responsesBlockFixture {
	ws := `{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"q"}}`
	codeCall := func(name, id string) string {
		return `{"type":"server_tool_use","id":"` + id + `","name":"` + name + `","input":{"code":"print(1)"}}`
	}
	mcp := `{"type":"mcp_tool_use","id":"mcptoolu_1","name":"f","server_name":"s","input":{}}`
	out := map[AnthropicContentBlockType]responsesBlockFixture{
		AnthropicContentBlockTypeText:                              {"", `{"type":"text","text":"hi"}`},
		AnthropicContentBlockTypeImage:                             {"", `{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgo="}}`},
		AnthropicContentBlockTypeDocument:                          {"", `{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0="}}`},
		AnthropicContentBlockTypeContainerUpload:                   {"", `{"type":"container_upload","file_id":"file_1"}`},
		AnthropicContentBlockTypeThinking:                          {"", `{"type":"thinking","thinking":"plan","signature":"sig"}`},
		AnthropicContentBlockTypeRedactedThinking:                  {"", `{"type":"redacted_thinking","data":"opaque"}`},
		AnthropicContentBlockTypeToolUse:                           {"", `{"type":"tool_use","id":"toolu_1","name":"f","input":{"a":1}}`},
		AnthropicContentBlockTypeToolResult:                        {`{"type":"tool_use","id":"toolu_1","name":"f","input":{}}`, `{"type":"tool_result","tool_use_id":"toolu_1","content":"ok"}`},
		AnthropicContentBlockTypeServerToolUse:                     {"", ws},
		AnthropicContentBlockTypeWebSearchToolResult:               {ws, `{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[{"type":"web_search_result","url":"https://a","title":"A","encrypted_content":"e"}]}`},
		AnthropicContentBlockTypeWebSearchToolResultError:          {ws, `{"type":"web_search_tool_result_error","tool_use_id":"srvtoolu_1","error_code":"unavailable"}`},
		AnthropicContentBlockTypeWebFetchToolResult:                {`{"type":"server_tool_use","id":"srvtoolu_2","name":"web_fetch","input":{"url":"https://a"}}`, `{"type":"web_fetch_tool_result","tool_use_id":"srvtoolu_2","content":{"type":"web_fetch_result","url":"https://a","content":{"type":"document","source":{"type":"text","media_type":"text/plain","data":"x"}}}}`},
		AnthropicContentBlockTypeCodeExecutionToolResult:           {codeCall("code_execution", "srvtoolu_3"), `{"type":"code_execution_tool_result","tool_use_id":"srvtoolu_3","content":{"type":"code_execution_result","stdout":"1","stderr":"","return_code":0,"content":[]}}`},
		AnthropicContentBlockTypeBashCodeExecutionToolResult:       {codeCall("bash_code_execution", "srvtoolu_4"), `{"type":"bash_code_execution_tool_result","tool_use_id":"srvtoolu_4","content":{"type":"bash_code_execution_result","stdout":"1","stderr":"","return_code":0,"content":[]}}`},
		AnthropicContentBlockTypeTextEditorCodeExecutionToolResult: {codeCall("text_editor_code_execution", "srvtoolu_5"), `{"type":"text_editor_code_execution_tool_result","tool_use_id":"srvtoolu_5","content":{"type":"text_editor_code_execution_result","file_type":"text","content":"x"}}`},
		AnthropicContentBlockTypeToolSearchToolResult:              {`{"type":"server_tool_use","id":"srvtoolu_6","name":"tool_search_tool_regex","input":{"query":"x"}}`, `{"type":"tool_search_tool_result","tool_use_id":"srvtoolu_6","content":{"type":"tool_search_tool_search_result","tool_references":[{"type":"tool_reference","tool_name":"f"}]}}`},
		AnthropicContentBlockTypeAdvisorToolResult:                 {`{"type":"server_tool_use","id":"srvtoolu_7","name":"advisor","input":{}}`, `{"type":"advisor_tool_result","tool_use_id":"srvtoolu_7","content":{"type":"advisor_result","text":"t"}}`},
		AnthropicContentBlockTypeMCPToolUse:                        {"", mcp},
		AnthropicContentBlockTypeMCPToolResult:                     {mcp, `{"type":"mcp_tool_result","tool_use_id":"mcptoolu_1","is_error":false,"content":[{"type":"text","text":"r"}]}`},
		AnthropicContentBlockTypeCompaction:                        {"", `{"type":"compaction","content":"summary"}`},
		AnthropicContentBlockTypeFallback:                          {"", `{"type":"fallback","from":{"model":"a"},"to":{"model":"b"}}`},
	}
	for _, ct := range ContentBlockTypes {
		if _, ok := out[ct]; !ok {
			out[ct] = responsesBlockFixture{"", unmappedBlock(ct)}
		}
	}
	return out
}

func answerWith(blocks ...string) string {
	var kept []string
	for _, b := range blocks {
		if b != "" {
			kept = append(kept, b)
		}
	}
	return unaryWith(kept...)
}

// convertResponsesAnswer decodes body and converts it, with the raw body handed over or not.
func convertResponsesAnswer(t *testing.T, body string, setRaw bool) (*schemas.BifrostResponsesResponse, *schemas.BifrostContext) {
	t.Helper()
	resp := AcquireAnthropicMessageResponse()
	t.Cleanup(func() { ReleaseAnthropicMessageResponse(resp) })
	if err := sonic.Unmarshal([]byte(body), resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if setRaw {
		resp.SetRawBody([]byte(body))
	}
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	return resp.ToBifrostResponsesResponse(ctx), ctx
}

// providerBlockItems returns the provider_block items of output, marshalled.
func providerBlockItems(t *testing.T, output []schemas.ResponsesMessage) []string {
	t.Helper()
	var out []string
	for i := range output {
		if output[i].Type != nil && *output[i].Type == schemas.ResponsesMessageTypeProviderBlock {
			b, err := output[i].MarshalJSON()
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, string(b))
		}
	}
	return out
}

func TestProviderBlocks_ResponsesMapsBlock(t *testing.T) {
	fixtures := responsesBlockFixtures()
	for _, ct := range ContentBlockTypes {
		t.Run(string(ct), func(t *testing.T) {
			fx := fixtures[ct]
			resp, ctx := convertResponsesAnswer(t, answerWith(fx.pre, fx.block), true)
			items := providerBlockItems(t, resp.Output)
			if ResponsesMapsBlock(ct) {
				if len(items) != 0 {
					t.Fatalf("mapped type yields provider blocks %v", items)
				}
				return
			}
			if len(items) != 1 {
				t.Fatalf("unmapped type yields %d provider blocks, want 1 (an answer arm exists?): %v", len(items), items)
			}
			if got, want := compactJSON(t, []byte(gjson.Get(items[0], "block").Raw)), compactJSON(t, []byte(fx.block)); got != want {
				t.Fatalf("block:\n got %s\nwant %s", got, want)
			}
			if gjson.Get(items[0], "provider").String() != schemas.ProviderBlockAnthropic || !strings.HasPrefix(gjson.Get(items[0], "id").String(), "pb_") {
				t.Fatalf("item envelope: %s", items[0])
			}
			if a := schemas.AdaptationsFromContext(ctx); a&schemas.AdaptationUnmapped != 0 {
				t.Fatalf("a carried block recorded %v", a)
			}
		})
	}
}

// outputTypes lists the output item types in order.
func outputTypes(output []schemas.ResponsesMessage) []string {
	out := make([]string, len(output))
	for i := range output {
		if output[i].Type != nil {
			out[i] = string(*output[i].Type)
		}
	}
	return out
}

func TestProviderBlocks_ResponsesUnplaced(t *testing.T) {
	for name, block := range map[string]string{
		"image without a source":    `{"type":"image"}`,
		"search error with no call": `{"type":"web_search_tool_result_error","tool_use_id":"srvtoolu_9","error_code":"unavailable"}`,
	} {
		t.Run(name, func(t *testing.T) {
			body := answerWith(`{"type":"text","text":"a"}`, block)
			resp, _ := convertResponsesAnswer(t, body, true)
			items := providerBlockItems(t, resp.Output)
			if len(items) != 1 || compactJSON(t, []byte(gjson.Get(items[0], "block").Raw)) != compactJSON(t, []byte(block)) {
				t.Fatalf("provider blocks %v, want the unplaced block", items)
			}
			if got := strings.Join(outputTypes(resp.Output), ","); got != "message,provider_block" {
				t.Fatalf("output %s", got)
			}

			resp, ctx := convertResponsesAnswer(t, body, false)
			if items := providerBlockItems(t, resp.Output); len(items) != 0 {
				t.Fatalf("no raw body, yet provider blocks %v", items)
			}
			if a := schemas.AdaptationsFromContext(ctx); a&schemas.AdaptationUnmapped == 0 {
				t.Fatalf("lost block recorded %v, want unmapped", a)
			}
		})
	}
}

// futureBlockAnswer is an answer with a block core has never seen between two texts.
const futureBlock = providerblocktest.FutureBlock

var futureBlockAnswer = unaryWith(`{"type":"text","text":"a"}`, futureBlock, `{"type":"text","text":"b"}`)

// streamedProviderBlock streams frames and returns the one provider_block done item and its
// copy in response.completed.
func streamedProviderBlock(t *testing.T, frames []string) (string, string) {
	t.Helper()
	events, last := responsesStreamFrom(t, frames)
	var done []string
	for _, e := range eventsOf(events, schemas.ResponsesStreamResponseTypeOutputItemDone) {
		if *e.Item.Type == schemas.ResponsesMessageTypeProviderBlock {
			done = append(done, itemJSON(t, e.Item))
		}
	}
	if len(done) != 1 {
		t.Fatalf("stream emitted %d provider_block items", len(done))
	}
	var completed []string
	for i := range last.Response.Output {
		if *last.Response.Output[i].Type == schemas.ResponsesMessageTypeProviderBlock {
			completed = append(completed, itemJSON(t, &last.Response.Output[i]))
		}
	}
	if len(completed) != 1 {
		t.Fatalf("response.completed holds %d provider_block items", len(completed))
	}
	return done[0], completed[0]
}

func TestProviderBlocks_ResponsesUnplacedStream(t *testing.T) {
	for name, tc := range map[string]struct {
		pre   string // a block ahead of the carried one
		block string
		delta string // an input_json_delta fragment, when the input streams
	}{
		"tool_use without an id":                  {block: `{"type":"tool_use","name":"f","input":{}}`},
		"known server tool without an id":         {block: `{"type":"server_tool_use","name":"web_search","input":{"query":"q"}}`},
		"web_search result for another call":      {pre: `{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"q"}}`, block: `{"type":"web_search_tool_result","tool_use_id":"srvtoolu_x","content":[]}`},
		"web_fetch result with no open call":      {block: `{"type":"web_fetch_tool_result","tool_use_id":"srvtoolu_x","content":{"type":"web_fetch_result","url":"https://a"}}`},
		"code execution result with no open call": {block: `{"type":"code_execution_tool_result","tool_use_id":"srvtoolu_x","content":{"type":"code_execution_result","stdout":"1","stderr":"","return_code":0,"content":[]}}`},
		"server tool without a Responses arm":     {block: `{"type":"server_tool_use","id":"srvtoolu_8","name":"future_tool","input":{}}`, delta: `{\"a\":1}`},
	} {
		t.Run(name, func(t *testing.T) {
			unaryBlock := tc.block
			if tc.delta != "" {
				unaryBlock = strings.Replace(tc.block, `"input":{}`, `"input":{"a":1}`, 1)
			}
			blocks := []string{tc.block}
			unaryBlocks := []string{unaryBlock}
			if tc.pre != "" {
				blocks = []string{tc.pre, tc.block}
				unaryBlocks = []string{tc.pre, unaryBlock}
			}
			unary, _ := convertResponsesAnswer(t, answerWith(unaryBlocks...), true)
			types := outputTypes(unary.Output)
			if len(types) == 0 || types[len(types)-1] != "provider_block" || strings.Count(strings.Join(types, ","), "provider_block") != 1 {
				t.Fatalf("unary output %v", types)
			}
			want := itemJSON(t, &unary.Output[len(unary.Output)-1])
			frames := streamFramesWith("", blocks...)
			if tc.delta != "" {
				frames = append(frames[:2], append([]string{`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"` + tc.delta + `"}}`}, frames[2:]...)...)
			}
			done, completed := streamedProviderBlock(t, frames)
			if done != want || completed != want {
				t.Fatalf("stream item\n done      %s\n completed %s\nwant       %s", done, completed, want)
			}
		})
	}

	t.Run("compaction with tool_changes and no summary is placed in both modes", func(t *testing.T) {
		block := `{"type":"compaction","tool_changes":[{"type":"tool_removal","tool":{"type":"function","name":"f"}}]}`
		unary, _ := convertResponsesAnswer(t, answerWith(block), true)
		if len(unary.Output) != 1 || gjson.Get(itemJSON(t, &unary.Output[0]), "content.0.tool_changes").Raw == "" {
			t.Fatalf("unary output %v, want the compaction with its tool_changes", outputTypes(unary.Output))
		}
		want := itemJSON(t, &unary.Output[0])
		events, last := responsesStreamFrom(t, streamFramesWith("", block))
		dones := eventsOf(events, schemas.ResponsesStreamResponseTypeOutputItemDone)
		if len(dones) != 1 || itemJSON(t, dones[0].Item) != want {
			t.Fatalf("stream done items %d, want one equal to\n%s", len(dones), want)
		}
		if len(last.Response.Output) != 1 || itemJSON(t, &last.Response.Output[0]) != want {
			t.Fatalf("completed output %v", outputTypes(last.Response.Output))
		}
	})

	t.Run("payload-free blocks yield no item", func(t *testing.T) {
		for _, block := range []string{`{"type":"redacted_thinking"}`, `{"type":"redacted_thinking","data":""}`, `{"type":"compaction"}`} {
			unary, ctx := convertResponsesAnswer(t, answerWith(block), true)
			if len(unary.Output) != 0 || schemas.AdaptationsFromContext(ctx) != 0 {
				t.Errorf("%s: unary output %v, adaptations %v", block, outputTypes(unary.Output), schemas.AdaptationsFromContext(ctx))
			}
			events, last := responsesStreamFrom(t, streamFramesWith("", block))
			if n := len(eventsOf(events, schemas.ResponsesStreamResponseTypeOutputItemDone)); n != 0 || len(last.Response.Output) != 0 {
				t.Errorf("%s: stream %d done events, completed %v", block, n, outputTypes(last.Response.Output))
			}
		}
	})

	t.Run("thinking is always a reasoning item", func(t *testing.T) {
		for _, block := range []string{`{"type":"thinking","signature":"sig"}`, `{"type":"thinking","thinking":"","signature":"sig"}`} {
			unary, _ := convertResponsesAnswer(t, answerWith(block), true)
			if got := strings.Join(outputTypes(unary.Output), ","); got != "reasoning" {
				t.Fatalf("%s: unary output %s", block, got)
			}
			want := itemJSON(t, &unary.Output[0])
			if gjson.Get(want, "content.0.signature").Str != "sig" {
				t.Fatalf("%s: unary item lost the signature: %s", block, want)
			}
			frames := streamFramesWith("", `{"type":"thinking","thinking":""}`)
			frames = append(frames[:2], append([]string{`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}`}, frames[2:]...)...)
			events, last := responsesStreamFrom(t, frames)
			dones := eventsOf(events, schemas.ResponsesStreamResponseTypeOutputItemDone)
			// The two converters already differ on a reasoning item's role and summary
			// members, for every thinking block; the reasoning content is what must agree.
			want = gjson.Get(want, "content").Raw
			if len(dones) != 1 || gjson.Get(itemJSON(t, dones[0].Item), "content").Raw != want ||
				gjson.Get(itemJSON(t, &last.Response.Output[0]), "content").Raw != want {
				t.Fatalf("%s: stream differs from unary\n want %s\n done %v", block, want, itemJSON(t, dones[0].Item))
			}
		}
	})

	t.Run("no event bytes records the loss", func(t *testing.T) {
		ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
		state := AcquireAnthropicResponsesStreamState()
		t.Cleanup(func() { ReleaseAnthropicResponsesStreamState(state) })
		for _, e := range decodeEvents(t, streamFramesWith("", futureBlock)) {
			_, _, _ = e.ev.ToBifrostResponsesStream(ctx, 0, state)
		}
		if schemas.AdaptationsFromContext(ctx)&schemas.AdaptationUnmapped == 0 {
			t.Fatal("a block the fold could not start is not recorded")
		}
	})
}

func TestProviderBlocks_ResponsesRawBodyCallers(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(futureBlockAnswer))
	}))
	t.Cleanup(server.Close)
	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	resp, bifrostErr := newTruncationTestProvider(server.URL).Responses(ctx, schemas.Key{Value: *schemas.NewSecretVar("k")},
		&schemas.BifrostResponsesRequest{Provider: schemas.Anthropic, Model: "claude-sonnet-4-6", Input: makeSimpleInput("hi")})
	if bifrostErr != nil {
		t.Fatalf("Responses: %s", bifrostErr.Error.Message)
	}
	if items := providerBlockItems(t, resp.Output); len(items) != 1 || compactJSON(t, []byte(gjson.Get(items[0], "block").Raw)) != futureBlock {
		t.Fatalf("provider blocks %v", items)
	}
}

// itemJSON marshals an item with its id removed, so items from two conversions compare.
func itemJSON(t *testing.T, m *schemas.ResponsesMessage) string {
	t.Helper()
	b, err := sonic.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	out, err := sjson.Delete(string(b), "id")
	if err != nil {
		t.Fatal(err)
	}
	return compactJSON(t, []byte(out))
}

func eventsOf(events []*schemas.BifrostResponsesStreamResponse, typ schemas.ResponsesStreamResponseType) []*schemas.BifrostResponsesStreamResponse {
	var out []*schemas.BifrostResponsesStreamResponse
	for _, e := range events {
		if e.Type == typ {
			out = append(out, e)
		}
	}
	return out
}

func TestProviderBlocks_ResponsesUnknownType(t *testing.T) {
	unary, _ := convertResponsesAnswer(t, futureBlockAnswer, true)
	if got := strings.Join(outputTypes(unary.Output), ","); got != "message,provider_block,message" {
		t.Fatalf("unary output %s", got)
	}
	want := itemJSON(t, &unary.Output[1])
	if got := compactJSON(t, []byte(gjson.Get(want, "block").Raw)); got != futureBlock {
		t.Fatalf("unary block %s", got)
	}

	frames := streamFramesWith("", `{"type":"text","text":""}`, futureBlock, `{"type":"text","text":""}`)
	// Text items reach response.completed only with text, so each text block gets a delta.
	frames = append(frames[:2], append([]string{`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"a"}}`}, frames[2:]...)...)
	frames = append(frames[:7], append([]string{`{"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"b"}}`}, frames[7:]...)...)
	events, last := responsesStreamFrom(t, frames)
	var added, done *schemas.BifrostResponsesStreamResponse
	for _, e := range events {
		if e.Item == nil || e.Item.Type == nil || *e.Item.Type != schemas.ResponsesMessageTypeProviderBlock {
			continue
		}
		switch e.Type {
		case schemas.ResponsesStreamResponseTypeOutputItemAdded:
			added = e
		case schemas.ResponsesStreamResponseTypeOutputItemDone:
			done = e
		}
	}
	if added == nil || done == nil {
		t.Fatal("stream emitted no provider_block added/done")
	}
	if *added.OutputIndex != 1 || *done.OutputIndex != 1 || done.SequenceNumber != added.SequenceNumber+1 {
		t.Fatalf("indexes: added %d/%d, done %d/%d", *added.OutputIndex, added.SequenceNumber, *done.OutputIndex, done.SequenceNumber)
	}
	if got := itemJSON(t, done.Item); got != want {
		t.Fatalf("streamed item:\n got %s\nwant %s", got, want)
	}
	completed := last.Response.Output
	if got := strings.Join(outputTypes(completed), ","); got != "message,provider_block,message" {
		t.Fatalf("completed output %s", got)
	}
	id := *added.Item.ID
	if *done.Item.ID != id || completed[1].ID == nil || *completed[1].ID != id || !strings.HasPrefix(id, "pb_") {
		t.Fatalf("ids differ: added %s, done %s, completed %v", id, *done.Item.ID, completed[1].ID)
	}
	b, _ := completed[1].MarshalJSON()
	if gjson.GetBytes(b, "id").String() != id {
		t.Fatalf("completed item bytes carry another id: %s", b)
	}
}

func TestProviderBlocks_ResponsesStreamParity(t *testing.T) {
	fixtures := responsesBlockFixtures()
	for _, ct := range []AnthropicContentBlockType{
		AnthropicContentBlockTypeImage, AnthropicContentBlockTypeDocument, AnthropicContentBlockTypeContainerUpload,
		AnthropicContentBlockTypeToolResult, AnthropicContentBlockTypeMCPToolResult, AnthropicContentBlockTypeFallback,
	} {
		t.Run(string(ct), func(t *testing.T) {
			fx := fixtures[ct]
			unary, _ := convertResponsesAnswer(t, answerWith(fx.pre, fx.block), true)
			want := itemJSON(t, &unary.Output[len(unary.Output)-1])

			var blocks []string
			if fx.pre != "" {
				blocks = append(blocks, fx.pre)
			}
			events, last := responsesStreamFrom(t, streamFramesWith("", append(blocks, fx.block)...))
			dones := eventsOf(events, schemas.ResponsesStreamResponseTypeOutputItemDone)
			if len(dones) == 0 {
				t.Fatal("no output_item.done")
			}
			if got := itemJSON(t, dones[len(dones)-1].Item); got != want {
				t.Fatalf("streamed done:\n got %s\nwant %s", got, want)
			}
			completed := last.Response.Output
			if got := itemJSON(t, &completed[len(completed)-1]); got != want {
				t.Fatalf("completed item:\n got %s\nwant %s", got, want)
			}
		})
	}

	t.Run("web_search_tool_result_error", func(t *testing.T) {
		fx := fixtures[AnthropicContentBlockTypeWebSearchToolResultError]
		unary, _ := convertResponsesAnswer(t, answerWith(fx.pre, fx.block), true)
		if len(unary.Output) != 1 || *unary.Output[0].Status != "failed" {
			t.Fatalf("unary output %+v", outputTypes(unary.Output))
		}
		events, last := responsesStreamFrom(t, streamFramesWith("", fx.pre, fx.block))
		var failed int
		for _, e := range eventsOf(events, schemas.ResponsesStreamResponseTypeOutputItemDone) {
			if *e.Item.Type == schemas.ResponsesMessageTypeWebSearchCall {
				if *e.Item.Status != "failed" {
					t.Fatalf("web_search_call done status %s", *e.Item.Status)
				}
				failed++
			}
		}
		if failed != 1 {
			t.Fatalf("%d web_search_call done events, want 1", failed)
		}
		completed := last.Response.Output
		if len(completed) != 1 || *completed[0].Status != "failed" {
			t.Fatalf("completed output %v", outputTypes(completed))
		}
	})

	t.Run("completed keeps every done item", func(t *testing.T) {
		mcp := responsesBlockFixtures()[AnthropicContentBlockTypeMCPToolUse].block
		_, last := responsesStreamFrom(t, streamFramesWith("", mcp, `{"type":"text","text":""}`))
		if got := strings.Join(outputTypes(last.Response.Output), ","); got != "mcp_call,message" {
			t.Fatalf("completed output %s, want the mcp_tool_use call and the empty text", got)
		}
		wf := responsesBlockFixtures()[AnthropicContentBlockTypeWebFetchToolResult]
		wfEvents, wfLast := responsesStreamFrom(t, streamFramesWith("", wf.pre, wf.block))
		wfDones := eventsOf(wfEvents, schemas.ResponsesStreamResponseTypeOutputItemDone)
		if got := strings.Join(outputTypes(wfLast.Response.Output), ","); got != "web_fetch_call" || len(wfDones) != 1 {
			t.Fatalf("web_fetch: completed output %s with %d done events, want the one web_fetch_call", got, len(wfDones))
		}
		if itemJSON(t, &wfLast.Response.Output[0]) != itemJSON(t, wfDones[0].Item) {
			t.Fatal("web_fetch: the completed web_fetch_call differs from its done item")
		}
		frames := streamFramesWith("", `{"type":"compaction"}`)
		frames = append(frames[:2], append([]string{`{"type":"content_block_delta","index":0,"delta":{"type":"compaction_delta","content":"sum"}}`}, frames[2:]...)...)
		events, last := responsesStreamFrom(t, frames)
		dones := eventsOf(events, schemas.ResponsesStreamResponseTypeOutputItemDone)
		if len(dones) != 1 || len(last.Response.Output) != 1 || itemJSON(t, &last.Response.Output[0]) != itemJSON(t, dones[0].Item) {
			t.Fatalf("compaction: %d done events, completed output %v; want its done item kept", len(dones), outputTypes(last.Response.Output))
		}
	})

	for _, ct := range []AnthropicContentBlockType{AnthropicContentBlockTypeContainerUpload, AnthropicContentBlockTypeFallback} {
		t.Run("delta and stop on a whole "+string(ct)+" block", func(t *testing.T) {
			frames := streamFramesWith("", fixtures[ct].block)
			plain, _ := responsesStreamFrom(t, frames)
			if n := len(eventsOf(plain, schemas.ResponsesStreamResponseTypeOutputItemDone)); n != 1 {
				t.Fatalf("%d output_item.done events, want the start's one", n)
			}
			withDelta := append(append(append([]string{}, frames[:2]...), `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x"}}`), frames[2:]...)
			events, last := responsesStreamFrom(t, withDelta)
			if len(events) != len(plain) {
				t.Fatalf("the delta emitted %d events", len(events)-len(plain))
			}
			if d := last.ExtraFields.ProviderBlockDrops; d.UnknownDelta != 1 || d.Interleaved != 0 {
				t.Fatalf("drops %+v, want one unknown delta", d)
			}
		})
	}
}

// replayItem is a provider_block input item; an empty provider or block member is absent.
func replayItem(provider, block string) string {
	item := `{"type":"provider_block"`
	if provider != "" {
		item += `,"provider":` + provider
	}
	if block != "" {
		item += `,"block":` + block
	}
	return item + `}`
}

func decodeItems(t *testing.T, items ...string) []schemas.ResponsesMessage {
	t.Helper()
	var out []schemas.ResponsesMessage
	if err := sonic.UnmarshalString("["+strings.Join(items, ",")+"]", &out); err != nil {
		t.Fatalf("decode items: %v", err)
	}
	return out
}

func buildReplayBody(t *testing.T, input []schemas.ResponsesMessage) ([]byte, *schemas.BifrostContext) {
	t.Helper()
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	body, bErr := BuildAnthropicResponsesRequestBody(ctx, &schemas.BifrostResponsesRequest{
		Provider: schemas.Anthropic, Model: "claude-sonnet-4-6", Input: input,
	}, AnthropicRequestBuildConfig{Provider: schemas.Anthropic, Model: "claude-sonnet-4-6"})
	if bErr != nil {
		t.Fatalf("build: %+v", bErr.Error)
	}
	return body, ctx
}

const (
	replayUser      = `{"type":"message","role":"user","content":"hi"}`
	replayAssistant = `{"type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"%s"}]}`
)

func TestProviderBlocks_ResponsesReplayConversion(t *testing.T) {
	// The rejection table is duplicated, row for row, in the router's
	// TestResponsesProviderBlocks_Verdict; a row change is made in both repos.
	t.Run("rejection table", func(t *testing.T) {
		rows := []struct {
			provider, block string
			accept          bool
		}{
			{`"anthropic"`, `{"type":"x"}`, true},
			{`"anthropic"`, `{"type":"x","cache_control":{"type":"ephemeral","ttl":"1h"}}`, true},
			{`"anthropic"`, `{"type":"x","cache_control":null}`, true},
			{`"openai"`, `{"type":"x"}`, false},
			{"", `{"type":"x"}`, false},
			{`1`, `{"type":"x"}`, false},
			{`"anthropic"`, `"x"`, false},
			{`"anthropic"`, `[{"type":"x"}]`, false},
			{`"anthropic"`, `null`, false},
			{`"anthropic"`, "", false},
			{`"anthropic"`, `{"type":"x","cache_control":"ephemeral"}`, false},
			{`"anthropic"`, `{"type":"x","cache_control":{"type":1}}`, false},
			{`"anthropic"`, `{"type":"x"}`, true},
		}
		for i, row := range rows {
			item := decodeItems(t, replayItem(row.provider, row.block))[0]
			provider, raw, ok := item.ProviderBlock()
			var block AnthropicContentBlock
			var err error
			if ok {
				block, err = replayedProviderBlock(provider, raw)
			}
			if got := ok && err == nil; got != row.accept {
				t.Errorf("row %d (%s): accept = %v, want %v", i+1, replayItem(row.provider, row.block), got, row.accept)
			}
			if i == 1 && (block.CacheControl == nil || block.CacheControl.TTL == nil || *block.CacheControl.TTL != "1h" || bytes.Contains(block.Raw, []byte("cache_control"))) {
				t.Errorf("row 2: cache_control not lifted: %+v %s", block.CacheControl, block.Raw)
			}
		}
	})

	t.Run("cache_control is typed and clamped", func(t *testing.T) {
		cached := `{"type":"message","role":"user","content":[{"type":"input_text","text":"%d","cache_control":{"type":"ephemeral"}}]}`
		items := []string{replayUser, replayItem(`"anthropic"`, `{"type":"tool_addition","x":1,"cache_control":{"type":"ephemeral"}}`)}
		for i := 0; i < AnthropicMaxCacheBreakpoints; i++ {
			items = append(items, fmt.Sprintf(cached, i))
		}
		body, ctx := buildReplayBody(t, decodeItems(t, items...))
		if n := strings.Count(string(body), `"cache_control"`); n != AnthropicMaxCacheBreakpoints {
			t.Fatalf("body carries %d cache_control markers, want %d: %s", n, AnthropicMaxCacheBreakpoints, body)
		}
		replayed := gjson.GetBytes(body, "messages.1.content.0")
		if replayed.Get("type").Str != "tool_addition" || replayed.Get("x").Int() != 1 || replayed.Get("cache_control").Exists() {
			t.Fatalf("the earliest marker, the replayed block's, was not the one cleared: %s", replayed.Raw)
		}
		if schemas.AdaptationsFromContext(ctx)&schemas.AdaptationUnmapped == 0 {
			t.Fatal("clamp not recorded")
		}
	})

	t.Run("uncarried items are left out and recorded", func(t *testing.T) {
		for name, item := range map[string]string{
			"foreign provider": replayItem(`"openai"`, `{"type":"x","zz":"marker"}`),
			"string block":     replayItem(`"anthropic"`, `"zzmarker"`),
		} {
			t.Run(name, func(t *testing.T) {
				body, ctx := buildReplayBody(t, decodeItems(t, replayUser, fmt.Sprintf(replayAssistant, "a"), item))
				if bytes.Contains(body, []byte("marker")) {
					t.Fatalf("the item reached the body: %s", body)
				}
				if schemas.AdaptationsFromContext(ctx)&schemas.AdaptationUnmapped == 0 {
					t.Fatal("the loss is not recorded")
				}
			})
		}
	})

	t.Run("block keeps its position in the assistant turn", func(t *testing.T) {
		const block = `{"type":"future_block","x":[1,"a"]}`
		body, _ := buildReplayBody(t, decodeItems(t, replayUser, fmt.Sprintf(replayAssistant, "a"),
			replayItem(`"anthropic"`, block), fmt.Sprintf(replayAssistant, "b"), `{"type":"message","role":"user","content":"next"}`))
		msgs := gjson.GetBytes(body, "messages").Array()
		if len(msgs) != 3 || msgs[1].Get("role").Str != "assistant" {
			t.Fatalf("messages: %s", gjson.GetBytes(body, "messages").Raw)
		}
		content := msgs[1].Get("content").Array()
		if len(content) != 3 || content[0].Get("text").Str != "a" || content[1].Raw != block || content[2].Get("text").Str != "b" {
			t.Fatalf("assistant turn: %s", msgs[1].Raw)
		}
	})

	t.Run("assistant texts without a replayed block keep their boundaries", func(t *testing.T) {
		body, _ := buildReplayBody(t, decodeItems(t, replayUser, fmt.Sprintf(replayAssistant, "a"), fmt.Sprintf(replayAssistant, "b"),
			`{"type":"message","role":"user","content":"next"}`))
		if n := len(gjson.GetBytes(body, "messages").Array()); n != 4 {
			t.Fatalf("messages: %s", gjson.GetBytes(body, "messages").Raw)
		}
	})

	t.Run("feature asks", func(t *testing.T) {
		for blockType, want := range map[string]Feature{
			"compaction":    FeatureCompactionBlock,
			"fallback":      FeatureFallbackBlock,
			"tool_addition": FeatureToolChangeBlock,
			"tool_removal":  FeatureToolChangeBlock,
		} {
			item := replayItem(`"anthropic"`, `{"type":"`+blockType+`"}`)
			req := &schemas.BifrostResponsesRequest{Input: decodeItems(t, replayUser, item)}
			if got := ResponsesAsk(req, nil).Features; got&want == 0 {
				t.Errorf("ResponsesAsk %s: %v, want %v", blockType, got, want)
			}
			if got := ResponsesBodyAsk([]byte(`{"model":"m","input":[`+replayUser+`,`+item+`]}`), nil).Features; got&want == 0 {
				t.Errorf("ResponsesBodyAsk %s: %v, want %v", blockType, got, want)
			}
		}
	})
}

// TestProviderBlocks_ResponsesUnplacedStartState has a row per not-placed start path
// (rejected type, guard, each open-call match, server tool without an arm): the typed arm writes nothing,
// the carry records only the block's own index, and the block then streams as a
// provider_block item.
func TestProviderBlocks_ResponsesUnplacedStartState(t *testing.T) {
	for _, tc := range []struct {
		name  string
		open  func(*AnthropicResponsesStreamState)
		block string
	}{
		{"type ResponsesMapsBlock rejects", nil, futureBlock},
		{"image without a source", nil, `{"type":"image"}`},
		{"web_fetch_tool_result, other open call", func(s *AnthropicResponsesStreamState) { s.WebFetchToolID = schemas.Ptr("srvtoolu_1") },
			`{"type":"web_fetch_tool_result","tool_use_id":"srvtoolu_x","content":{"type":"x"}}`},
		{"server_tool_use with an unknown name", nil, `{"type":"server_tool_use","id":"srvtoolu_8","name":"future_tool","input":{}}`},
		{"server_tool_use without an id", nil, `{"type":"server_tool_use","name":"web_search","input":{"query":"q"}}`},
		{"web_search_tool_result, other open call", func(s *AnthropicResponsesStreamState) { s.WebSearchToolID = schemas.Ptr("srvtoolu_1") },
			`{"type":"web_search_tool_result","tool_use_id":"srvtoolu_x","content":[]}`},
		{"web_search_tool_result_error, other open call", func(s *AnthropicResponsesStreamState) { s.WebSearchToolID = schemas.Ptr("srvtoolu_1") },
			`{"type":"web_search_tool_result_error","tool_use_id":"srvtoolu_x","error_code":"unavailable"}`},
		{"tool_search_tool_result, other open call", func(s *AnthropicResponsesStreamState) { s.ToolSearchToolID = schemas.Ptr("srvtoolu_1") },
			`{"type":"tool_search_tool_result","tool_use_id":"srvtoolu_x","content":{"type":"x"}}`},
		{"advisor_tool_result, other open call", func(s *AnthropicResponsesStreamState) { s.AdvisorToolID = schemas.Ptr("srvtoolu_1") },
			`{"type":"advisor_tool_result","tool_use_id":"srvtoolu_x","content":{"type":"x"}}`},
		{"code_execution_tool_result, other open call", func(s *AnthropicResponsesStreamState) { s.CodeExecToolID = schemas.Ptr("srvtoolu_1") },
			`{"type":"code_execution_tool_result","tool_use_id":"srvtoolu_x","content":{"type":"x"}}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
			state := AcquireAnthropicResponsesStreamState()
			t.Cleanup(func() { ReleaseAnthropicResponsesStreamState(state) })
			if tc.open != nil {
				tc.open(state)
			}
			evs := decodeEvents(t, []string{
				`{"type":"content_block_start","index":0,"content_block":` + tc.block + `}`,
				`{"type":"content_block_stop","index":0}`,
			})
			blockTypes, outputIndexes := maps.Clone(state.ContentIndexToBlockType), maps.Clone(state.ContentIndexToOutputIndex)
			items := len(state.OutputItems)
			results := [...]*AnthropicContentBlock{state.WebSearchResult, state.WebFetchResult, state.AdvisorResult, state.CodeExecResult, state.ToolSearchResult}

			if out, _, _ := evs[0].ev.toBifrostResponsesStream(ctx, 0, state, evs[0].data); len(out) != 0 {
				t.Fatalf("an unplaced start emitted %d events", len(out))
			}
			// The carry owns index 0 now; everything else is as before the start.
			delete(state.ContentIndexToBlockType, 0)
			delete(state.ContentIndexToOutputIndex, 0)
			if !maps.Equal(state.ContentIndexToBlockType, blockTypes) || !maps.Equal(state.ContentIndexToOutputIndex, outputIndexes) || len(state.OutputItems) != items {
				t.Fatalf("state written: block types %v -> %v, output indexes %v -> %v, items %d -> %d",
					blockTypes, state.ContentIndexToBlockType, outputIndexes, state.ContentIndexToOutputIndex, items, len(state.OutputItems))
			}
			if got := [...]*AnthropicContentBlock{state.WebSearchResult, state.WebFetchResult, state.AdvisorResult, state.CodeExecResult, state.ToolSearchResult}; got != results {
				t.Fatalf("a stored result changed: %v -> %v", results, got)
			}

			out, _, _ := evs[1].ev.toBifrostResponsesStream(ctx, 1, state, evs[1].data)
			done := eventsOf(out, schemas.ResponsesStreamResponseTypeOutputItemDone)
			if len(done) != 1 || *done[0].Item.Type != schemas.ResponsesMessageTypeProviderBlock {
				t.Fatalf("the block did not stream as a provider_block item: %d done events", len(done))
			}
			if _, block, _ := done[0].Item.ProviderBlock(); compactJSON(t, block) != compactJSON(t, []byte(tc.block)) {
				t.Fatalf("carried block %s, want %s", block, tc.block)
			}
		})
	}
}

// Measured on the base pin: an answer whose blocks are all mapped and placed, and a text
// delta, allocate no more than before provider blocks reached Responses.
const (
	responsesMappedAnswerAllocs = 28
	responsesTextDeltaAllocs    = 4
)

const responsesMappedAnswer = `{"id":"m","type":"message","role":"assistant","model":"claude","stop_reason":"tool_use","content":[{"type":"text","text":"Hi"},{"type":"tool_use","id":"t","name":"f","input":{"a":1}}],"usage":{"input_tokens":1,"output_tokens":2}}`

func TestToBifrostResponsesResponse_MappedAnswer_AllocBudget(t *testing.T) {
	skipAllocUnderRace(t)
	body := []byte(responsesMappedAnswer)
	var resp AnthropicMessageResponse
	if err := sonic.Unmarshal(body, &resp); err != nil {
		t.Fatal(err)
	}
	resp.SetRawBody(body)
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	got := testing.AllocsPerRun(200, func() { _ = resp.ToBifrostResponsesResponse(ctx) })
	t.Logf("mapped text-plus-tool answer: %v allocs", got)
	if got > responsesMappedAnswerAllocs {
		t.Fatalf("a mapped answer allocates %v, budget %d", got, responsesMappedAnswerAllocs)
	}
}

func TestToBifrostResponsesStream_TextAnswer_AllocBudget(t *testing.T) {
	skipAllocUnderRace(t)
	setup := decodeEvents(t, []string{
		`{"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","model":"claude-sonnet-4-5","content":[],"usage":{"input_tokens":12,"output_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
	})
	delta := decodeEvents(t, []string{`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"word "}}`})[0]
	state := AcquireAnthropicResponsesStreamState()
	t.Cleanup(func() { ReleaseAnthropicResponsesStreamState(state) })
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	for _, e := range setup {
		_, _, _ = e.ev.toBifrostResponsesStream(ctx, 0, state, e.data)
	}
	got := testing.AllocsPerRun(200, func() {
		_, _, _ = delta.ev.toBifrostResponsesStream(ctx, 0, state, delta.data)
	})
	t.Logf("text delta: %v allocs", got)
	if got > responsesTextDeltaAllocs {
		t.Fatalf("a text delta allocates %v, budget %d", got, responsesTextDeltaAllocs)
	}
}

func BenchmarkToBifrostResponsesResponse_MappedAnswer(b *testing.B) {
	body := []byte(responsesMappedAnswer)
	var resp AnthropicMessageResponse
	if err := sonic.Unmarshal(body, &resp); err != nil {
		b.Fatal(err)
	}
	resp.SetRawBody(body)
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	b.ReportAllocs()
	for b.Loop() {
		_ = resp.ToBifrostResponsesResponse(ctx)
	}
}

func BenchmarkToBifrostResponsesStream_TextDelta(b *testing.B) {
	events := make([]decodedEvent, 0, 3)
	for _, s := range []string{
		`{"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","model":"claude-sonnet-4-5","content":[],"usage":{"input_tokens":12,"output_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"word "}}`,
	} {
		e := decodedEvent{ev: &AnthropicStreamEvent{}, data: []byte(s)}
		if err := sonic.Unmarshal(e.data, e.ev); err != nil {
			b.Fatal(err)
		}
		events = append(events, e)
	}
	state := AcquireAnthropicResponsesStreamState()
	b.Cleanup(func() { ReleaseAnthropicResponsesStreamState(state) })
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	for _, e := range events[:2] {
		_, _, _ = e.ev.toBifrostResponsesStream(ctx, 0, state, e.data)
	}
	delta := events[2]
	b.ReportAllocs()
	for b.Loop() {
		_, _, _ = delta.ev.toBifrostResponsesStream(ctx, 0, state, delta.data)
	}
}

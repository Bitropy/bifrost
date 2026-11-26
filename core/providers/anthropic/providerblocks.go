package anthropic

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"unsafe"

	"github.com/bytedance/sonic"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/tidwall/gjson"
)

// bytesAsString views b as a string for gjson without copying; the result must not outlive b.
func bytesAsString(b []byte) string {
	return unsafe.String(unsafe.SliceData(b), len(b))
}

// appendRawProviderCitations appends a provider_citation annotation for each provider
// citation of one text block, with the citation object copied from the block's raw bytes.
func appendRawProviderCitations(out []schemas.ChatAssistantMessageAnnotation, cs []AnthropicTextCitation, raw gjson.Result, start, end int) []schemas.ChatAssistantMessageAnnotation {
	if !hasProviderCitation(cs) {
		return out
	}
	span := &[2]int{start, end}
	m := 0
	raw.Get("citations").ForEach(func(_, c gjson.Result) bool {
		if m < len(cs) && isProviderCitation(cs[m]) && c.IsObject() {
			out = append(out, providerCitationAnnotation(span, json.RawMessage(c.Raw)))
		}
		m++
		return true
	})
	return out
}

// isProviderCitation reports whether a citation has no url_citation mapping and reaches
// the client as a provider_citation annotation.
func isProviderCitation(c AnthropicTextCitation) bool {
	return c.Type != AnthropicCitationTypeWebSearchResultLocation
}

func hasProviderCitation(cs []AnthropicTextCitation) bool {
	for i := range cs {
		if isProviderCitation(cs[i]) {
			return true
		}
	}
	return false
}

// providerCitationAnnotation is one provider citation spanning a text block. A block's
// annotations share one span; unlike url_citation, nothing is de-duplicated.
func providerCitationAnnotation(span *[2]int, citation json.RawMessage) schemas.ChatAssistantMessageAnnotation {
	return schemas.ChatAssistantMessageAnnotation{
		Type:       schemas.ChatAnnotationTypeProviderCitation,
		Provider:   schemas.ProviderBlockAnthropic,
		StartIndex: &span[0],
		EndIndex:   &span[1],
		Citation:   citation,
	}
}

// keepProviderCitations buffers the raw objects of a text block's provider citations from
// the content_block_start event's citation array.
func (state *AnthropicStreamState) keepProviderCitations(blockIndex int, cs []AnthropicTextCitation) {
	if !hasProviderCitation(cs) {
		return
	}
	data := state.eventData
	bs, be := jsonMember(data, 0, len(data), "content_block")
	as, ae := jsonMember(data, bs, be, "citations")
	m := 0
	jsonEachElem(data, as, ae, func(s, e int) {
		if m < len(cs) && isProviderCitation(cs[m]) {
			state.keepRawCitation(blockIndex, s, e)
		}
		m++
	})
}

// keepDeltaProviderCitation buffers the raw object of a citations_delta's citation.
func (state *AnthropicStreamState) keepDeltaProviderCitation(blockIndex int) {
	data := state.eventData
	ds, de := jsonMember(data, 0, len(data), "delta")
	s, e := jsonMember(data, ds, de, "citation")
	state.keepRawCitation(blockIndex, s, e)
}

// keepRawCitation copies the citation object at eventData[s:e].
func (state *AnthropicStreamState) keepRawCitation(blockIndex, s, e int) {
	if !isJSONObjectAt(state.eventData, s, e) {
		return
	}
	if state.pendingCitationRaw == nil {
		state.pendingCitationRaw = make(map[int][]json.RawMessage)
	}
	raw := append(json.RawMessage(nil), state.eventData[s:e]...)
	state.pendingCitationRaw[blockIndex] = append(state.pendingCitationRaw[blockIndex], raw)
}

// providerBlockEvent handles every event of a chat-unmapped block, ahead of the typed
// conversion, and reports whether it did.
func (state *AnthropicStreamState) providerBlockEvent(chunk *AnthropicStreamEvent) (*schemas.BifrostChatResponse, bool) {
	if chunk.Type == AnthropicStreamEventTypeContentBlockStart {
		if chunk.ContentBlock == nil || chunk.Index == nil {
			return nil, false
		}
		state.interrupt()
		if ChatMapsBlock(chunk.ContentBlock.Type) {
			return nil, false
		}
		if state.start(*chunk.Index) {
			// The block's index comes from the counter reasoning details share with provider blocks.
			state.openShared = state.nextReasoningDetailIdx
			state.nextReasoningDetailIdx++
		}
		return nil, true
	}
	block, st := state.event(chunk)
	switch st {
	case foldConsumed:
		return nil, true
	case foldClosed:
		blocks := []schemas.ChatProviderBlock{{Index: state.openShared, Provider: schemas.ProviderBlockAnthropic, Block: block}}
		return &schemas.BifrostChatResponse{
			Object: "chat.completion.chunk",
			Choices: []schemas.BifrostResponseChoice{{
				Index: 0,
				ChatStreamResponseChoice: &schemas.ChatStreamResponseChoice{
					Delta: &schemas.ChatStreamResponseChoiceDelta{ProviderBlocks: &blocks},
				},
			}},
		}, true
	}
	return nil, false
}

// anthropicReasoningBlock maps one replayed reasoning detail to its Anthropic block.
func anthropicReasoningBlock(detail schemas.ChatReasoningDetails) AnthropicContentBlock {
	// reasoning.encrypted details carrying data hold an anthropic redacted_thinking
	// payload; replay the block as-is so the API can decrypt it. Encrypted details
	// without data (e.g. gemini thought signatures) keep the thinking-block mapping.
	if detail.Type == schemas.BifrostReasoningDetailsTypeEncrypted && detail.Data != nil && *detail.Data != "" {
		return AnthropicContentBlock{Type: AnthropicContentBlockTypeRedactedThinking, Data: detail.Data}
	}
	thinking := detail.Text
	// Only Anthropic-shaped text details: a Gemini thought signature (encrypted, no
	// data) must stay text-nil so the strip removes it; a foreign signature is a 400.
	if detail.Type == schemas.BifrostReasoningDetailsTypeText {
		thinking = replayedThinkingText(detail.Text, detail.Signature)
	}
	return AnthropicContentBlock{Type: AnthropicContentBlockTypeThinking, Signature: detail.Signature, Thinking: thinking}
}

// appendAnthropicLeadingRun appends an assistant turn's reasoning and replayed provider
// blocks to dst, merged by their shared index; on a tie reasoning comes first. msgIndex only
// names the entry in an error, which never echoes client bytes.
func appendAnthropicLeadingRun(dst []AnthropicContentBlock, msgIndex int, msg *schemas.ChatAssistantMessage) ([]AnthropicContentBlock, error) {
	if len(msg.ProviderBlocks) == 0 {
		for _, detail := range msg.ReasoningDetails {
			dst = append(dst, anthropicReasoningBlock(detail))
		}
		return dst, nil
	}
	type entry struct {
		index int
		block AnthropicContentBlock
	}
	entries := make([]entry, 0, len(msg.ReasoningDetails)+len(msg.ProviderBlocks))
	for _, detail := range msg.ReasoningDetails {
		entries = append(entries, entry{detail.Index, anthropicReasoningBlock(detail)})
	}
	for k, pb := range msg.ProviderBlocks {
		block, err := replayedProviderBlock(pb.Provider, pb.Block)
		if err != nil {
			return nil, fmt.Errorf("messages[%d].provider_blocks[%d]: %w", msgIndex, k, err)
		}
		entries = append(entries, entry{pb.Index, block})
	}
	sort.SliceStable(entries, func(a, b int) bool { return entries[a].index < entries[b].index })
	for i := range entries {
		dst = append(dst, entries[i].block)
	}
	return dst, nil
}

// replayedProviderBlock turns a replayed entry into a raw content block. Its type and
// cache_control are lifted into the typed fields, so every typed strip and gate reads the
// block as it reads a typed one, and cache_control has one home.
// It reads decoded values, so an escaped provider name passes here; the router's check reads
// the raw bytes and refuses it first. The router is the stricter on purpose: do not align it.
func replayedProviderBlock(provider string, block json.RawMessage) (AnthropicContentBlock, error) {
	if provider != schemas.ProviderBlockAnthropic {
		return AnthropicContentBlock{}, fmt.Errorf("provider is not %q", schemas.ProviderBlockAnthropic)
	}
	raw := bytes.TrimSpace(block)
	if len(raw) == 0 || raw[0] != '{' {
		return AnthropicContentBlock{}, fmt.Errorf("block is not a JSON object")
	}
	out := AnthropicContentBlock{Raw: json.RawMessage(raw)}
	if t := gjson.GetBytes(raw, "type"); t.Type == gjson.String {
		out.Type = AnthropicContentBlockType(t.Str)
	}
	if cc := gjson.GetBytes(raw, "cache_control"); cc.Exists() {
		if cc.Type != gjson.Null {
			var c schemas.CacheControl
			if err := sonic.UnmarshalString(cc.Raw, &c); err != nil {
				return AnthropicContentBlock{}, fmt.Errorf("cache_control is not a cache control object")
			}
			out.CacheControl = &c
		}
		stripped, err := providerUtils.DeleteJSONField(raw, "cache_control")
		if err != nil {
			return AnthropicContentBlock{}, fmt.Errorf("cache_control could not be moved")
		}
		out.Raw = stripped
	}
	return out, nil
}

func hasRawContentBlock(blocks []AnthropicContentBlock) bool {
	for i := range blocks {
		if blocks[i].Raw != nil {
			return true
		}
	}
	return false
}

// marshalContentBlocksWithRaw writes typed blocks as MarshalSorted does and replayed blocks
// verbatim, with the typed cache_control as a replayed block's last member.
func marshalContentBlocksWithRaw(blocks []AnthropicContentBlock) ([]byte, error) {
	buf := []byte{'['}
	for i := range blocks {
		if i > 0 {
			buf = append(buf, ',')
		}
		b := &blocks[i]
		if b.Raw == nil {
			data, err := providerUtils.MarshalSorted(b)
			if err != nil {
				return nil, err
			}
			buf = append(buf, data...)
			continue
		}
		if b.CacheControl == nil {
			buf = append(buf, b.Raw...)
			continue
		}
		cc, err := providerUtils.MarshalSorted(b.CacheControl)
		if err != nil {
			return nil, err
		}
		// Raw never holds cache_control, so sjson appends it as the last member.
		data, err := providerUtils.SetRawJSONField(b.Raw, "cache_control", cc)
		if err != nil {
			return nil, err
		}
		buf = append(buf, data...)
	}
	return append(buf, ']'), nil
}

// A minimal JSON span scanner for the stream converter's provider-block and
// provider-citation paths. It returns byte offsets, never slices or strings, so
// escape analysis sees no flow from its input: the stream state hands it the event
// bytes, and a scanner that returned views into them (gjson does) would move the
// whole state, its maps included, to the heap on every stream. It assumes JSON the
// upstream produced and only finds value boundaries; it does not validate.

// jsonSkipWS returns the offset of the first non-whitespace byte at or after i.
func jsonSkipWS(b []byte, i int) int {
	for i < len(b) && (b[i] == ' ' || b[i] == '\t' || b[i] == '\n' || b[i] == '\r') {
		i++
	}
	return i
}

// jsonValueEnd returns the offset just past the value starting at i, or -1.
func jsonValueEnd(b []byte, i int) int {
	if i >= len(b) {
		return -1
	}
	switch b[i] {
	case '"':
		for j := i + 1; j < len(b); j++ {
			switch b[j] {
			case '\\':
				j++
			case '"':
				return j + 1
			}
		}
		return -1
	case '{', '[':
		depth := 0
		for j := i; j < len(b); j++ {
			switch b[j] {
			case '"':
				end := jsonValueEnd(b, j)
				if end < 0 {
					return -1
				}
				j = end - 1
			case '{', '[':
				depth++
			case '}', ']':
				depth--
				if depth == 0 {
					return j + 1
				}
			}
		}
		return -1
	}
	j := i
	for j < len(b) && b[j] != ',' && b[j] != '}' && b[j] != ']' && b[j] != ' ' && b[j] != '\t' && b[j] != '\n' && b[j] != '\r' {
		j++
	}
	if j == i {
		return -1
	}
	return j
}

// jsonMember returns the span of the value of key in the object b[start:end], or
// (-1, -1). Keys are compared as written, which is exact for the unescaped keys
// Anthropic uses.
func jsonMember(b []byte, start, end int, key string) (int, int) {
	if start < 0 || end > len(b) {
		return -1, -1
	}
	b = b[:end]
	i := jsonSkipWS(b, start)
	if i >= len(b) || b[i] != '{' {
		return -1, -1
	}
	i = jsonSkipWS(b, i+1)
	for i < len(b) && b[i] == '"' {
		kEnd := jsonValueEnd(b, i)
		if kEnd < 0 {
			return -1, -1
		}
		match := string(b[i+1:kEnd-1]) == key
		i = jsonSkipWS(b, kEnd)
		if i >= len(b) || b[i] != ':' {
			return -1, -1
		}
		vStart := jsonSkipWS(b, i+1)
		vEnd := jsonValueEnd(b, vStart)
		if vEnd < 0 {
			return -1, -1
		}
		if match {
			return vStart, vEnd
		}
		i = jsonSkipWS(b, vEnd)
		if i < len(b) && b[i] == ',' {
			i = jsonSkipWS(b, i+1)
		}
	}
	return -1, -1
}

// jsonEachElem calls fn with the span of each element of the array b[start:end].
func jsonEachElem(b []byte, start, end int, fn func(s, e int)) {
	if start < 0 || end > len(b) {
		return
	}
	b = b[:end]
	i := jsonSkipWS(b, start)
	if i >= len(b) || b[i] != '[' {
		return
	}
	i = jsonSkipWS(b, i+1)
	for i < len(b) && b[i] != ']' {
		e := jsonValueEnd(b, i)
		if e < 0 {
			return
		}
		fn(i, e)
		i = jsonSkipWS(b, e)
		if i < len(b) && b[i] == ',' {
			i = jsonSkipWS(b, i+1)
		}
	}
}

// isJSONObjectAt reports whether a value span holds an object.
func isJSONObjectAt(b []byte, s, e int) bool {
	return s >= 0 && e > s && b[s] == '{'
}

// replayedThinkingText is the thinking text a replayed block is sent with. A signed
// block without text is Anthropic's omitted display, which the client's stream
// accumulation leaves without a text member; it is sent as "" so the strip keeps
// it and its encrypted reasoning reaches the model.
//
// Pass only Anthropic-shaped sources: a foreign signature sent to Claude is a 400.
// Chat details are gated on reasoning.text by the caller; on Responses only Claude
// (Anthropic, Bedrock) puts a Signature on a reasoning_text block, Gemini's thought
// signature travels as encrypted content.
func replayedThinkingText(text, signature *string) *string {
	if text == nil && signature != nil && *signature != "" {
		return &emptyThinkingText
	}
	return text
}

// emptyThinkingText is shared so the omitted-display replay allocates nothing; the
// block is marshalled, never written through.
var emptyThinkingText = ""

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

// countUnmappedBlocks counts the blocks that become provider blocks, so the converter
// sizes ProviderBlocks once and walks the raw body only when there is one.
func countUnmappedBlocks(blocks []AnthropicContentBlock) int {
	n := 0
	for i := range blocks {
		if !ChatMapsBlock(blocks[i].Type) {
			n++
		}
	}
	return n
}

// fillRawResponseBytes copies, in one walk of body, the raw bytes of each unmapped
// content[j] into the matching provider block and of each provider citation into its
// waiting annotation, both in content order. The typed values are never re-marshalled:
// members core does not declare would be lost. Entries the body has no bytes for are
// dropped rather than sent empty.
func fillRawResponseBytes(body []byte, blocks []AnthropicContentBlock, pbs []schemas.ChatProviderBlock, anns []schemas.ChatAssistantMessageAnnotation) ([]schemas.ChatProviderBlock, []schemas.ChatAssistantMessageAnnotation) {
	j, k, a := 0, 0, 0
	gjson.Get(bytesAsString(body), "content").ForEach(func(_, v gjson.Result) bool {
		if j >= len(blocks) {
			return false
		}
		b := &blocks[j]
		j++
		if !ChatMapsBlock(b.Type) {
			if k < len(pbs) && v.IsObject() {
				pbs[k].Block = json.RawMessage(v.Raw)
				k++
			}
			return true
		}
		if b.Type != AnthropicContentBlockTypeText || b.Text == nil || b.Citations == nil || !hasProviderCitation(b.Citations.TextCitations) {
			return true
		}
		m := 0
		v.Get("citations").ForEach(func(_, c gjson.Result) bool {
			if m < len(b.Citations.TextCitations) && isProviderCitation(b.Citations.TextCitations[m]) && c.IsObject() {
				for a < len(anns) && (anns[a].Type != schemas.ChatAnnotationTypeProviderCitation || anns[a].Citation != nil) {
					a++
				}
				if a < len(anns) {
					anns[a].Citation = json.RawMessage(c.Raw)
				}
			}
			m++
			return true
		})
		return true
	})
	kept := anns[:0]
	for _, ann := range anns {
		if ann.Type != schemas.ChatAnnotationTypeProviderCitation || ann.Citation != nil {
			kept = append(kept, ann)
		}
	}
	return pbs[:k], kept
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

// appendProviderCitationAnnotations appends one provider_citation annotation per provider
// citation of a text block spanning [start, end): from typed citations (unary, Citation
// filled later from the raw body) or from raw ones (stream). Unlike url_citation, each
// upstream citation is its own annotation; nothing is de-duplicated.
func appendProviderCitationAnnotations(out []schemas.ChatAssistantMessageAnnotation, typed []AnthropicTextCitation, raw []json.RawMessage, start, end int) ([]schemas.ChatAssistantMessageAnnotation, bool) {
	n := len(raw)
	for i := range typed {
		if isProviderCitation(typed[i]) {
			n++
		}
	}
	if n == 0 {
		return out, false
	}
	// One span pair per block, shared by its annotations.
	span := [2]int{start, end}
	add := func(citation json.RawMessage) {
		out = append(out, schemas.ChatAssistantMessageAnnotation{
			Type:       schemas.ChatAnnotationTypeProviderCitation,
			Provider:   schemas.ProviderBlockAnthropic,
			StartIndex: &span[0],
			EndIndex:   &span[1],
			Citation:   citation,
		})
	}
	for i := range typed {
		if isProviderCitation(typed[i]) {
			add(nil)
		}
	}
	for _, c := range raw {
		add(c)
	}
	return out, true
}

// keepProviderCitations buffers the raw objects of a text block's provider citations from
// the current event's citation array at path.
func (state *AnthropicStreamState) keepProviderCitations(blockIndex int, cs []AnthropicTextCitation, path string) {
	if !hasProviderCitation(cs) {
		return
	}
	m := 0
	gjson.Get(bytesAsString(state.eventData), path).ForEach(func(_, c gjson.Result) bool {
		if m < len(cs) && isProviderCitation(cs[m]) {
			state.keepRawCitation(blockIndex, c)
		}
		m++
		return true
	})
}

// keepProviderCitation buffers the raw object of one provider citation at path.
func (state *AnthropicStreamState) keepProviderCitation(blockIndex int, path string) {
	state.keepRawCitation(blockIndex, gjson.Get(bytesAsString(state.eventData), path))
}

func (state *AnthropicStreamState) keepRawCitation(blockIndex int, c gjson.Result) {
	if !c.IsObject() {
		return
	}
	if state.pendingCitationRaw == nil {
		state.pendingCitationRaw = make(map[int][]json.RawMessage)
	}
	state.pendingCitationRaw[blockIndex] = append(state.pendingCitationRaw[blockIndex], json.RawMessage(c.Raw))
}

// Stream folding: an unmapped block is buffered from its content_block_start and emitted
// whole at its content_block_stop. The scratch holds the start object followed by the
// value its deltas accumulate.
const (
	providerBlockClosed uint8 = iota
	providerBlockOpen
	providerBlockFoldInput   // input_json_delta: partial_json fragments of "input"
	providerBlockFoldContent // compaction_delta: escaped string fragments of "content"
)

type openProviderBlock struct {
	index, shared int
	kind          uint8
	objLen        int
}

// startProviderBlock opens an unmapped block from the event's raw content_block. Without
// the event bytes there is nothing verbatim to copy, so the block is not tracked at all.
func (state *AnthropicStreamState) startProviderBlock(blockIndex int) {
	raw := gjson.Get(bytesAsString(state.eventData), "content_block")
	if !raw.IsObject() {
		return
	}
	state.openBlock = openProviderBlock{index: blockIndex, shared: state.nextReasoningDetailIdx, kind: providerBlockOpen, objLen: len(raw.Raw)}
	state.nextReasoningDetailIdx++
	state.scratch = append(state.scratch[:0], raw.Raw...)
}

// foldProviderBlockDelta appends one delta of the open block to the scratch. A delta it
// cannot fold, or one that would fold into a second member, is dropped and counted.
func (state *AnthropicStreamState) foldProviderBlockDelta(delta *AnthropicStreamDelta) {
	var kind uint8
	var part string
	switch delta.Type {
	case AnthropicStreamDeltaTypeInputJSON:
		if delta.PartialJSON != nil {
			kind, part = providerBlockFoldInput, *delta.PartialJSON
		}
	case AnthropicStreamDeltaTypeCompaction:
		// The escaped bytes are kept as sent, so the fragments concatenate into one JSON string.
		if v := gjson.Get(bytesAsString(state.eventData), "delta.content"); v.Type == gjson.String {
			kind, part = providerBlockFoldContent, v.Raw[1:len(v.Raw)-1]
		}
	}
	if kind == providerBlockClosed || (state.openBlock.kind != providerBlockOpen && state.openBlock.kind != kind) {
		state.providerBlockDrops.UnknownDelta++
		return
	}
	state.openBlock.kind = kind
	state.scratch = append(state.scratch, part...)
}

// closeProviderBlock assembles the open block and returns its one chunk: the start object
// with the folded member replaced by the accumulated value, in a single allocation.
func (state *AnthropicStreamState) closeProviderBlock() *schemas.BifrostChatResponse {
	open := state.openBlock
	state.openBlock = openProviderBlock{}
	obj, acc := state.scratch[:open.objLen], state.scratch[open.objLen:]

	var member, quote string
	switch open.kind {
	case providerBlockFoldInput:
		member = "input"
	case providerBlockFoldContent:
		member, quote = "content", `"`
	}
	var block []byte
	if member == "" || len(acc) == 0 {
		block = bytes.Clone(obj)
	} else if v := gjson.Get(bytesAsString(obj), member); v.Exists() && v.Index > 0 {
		block = make([]byte, 0, len(obj)-len(v.Raw)+len(acc)+2*len(quote))
		block = append(block, obj[:v.Index]...)
		block = append(block, quote...)
		block = append(block, acc...)
		block = append(block, quote...)
		block = append(block, obj[v.Index+len(v.Raw):]...)
	} else {
		end := bytes.LastIndexByte(obj, '}')
		sep := ","
		if len(bytes.TrimSpace(obj[1:end])) == 0 {
			sep = ""
		}
		block = make([]byte, 0, len(obj)+len(sep)+len(member)+3+len(acc)+2*len(quote))
		block = append(block, obj[:end]...)
		block = append(block, sep...)
		block = append(block, '"')
		block = append(block, member...)
		block = append(block, `":`...)
		block = append(block, quote...)
		block = append(block, acc...)
		block = append(block, quote...)
		block = append(block, obj[end:]...)
	}

	return &schemas.BifrostChatResponse{
		Object: "chat.completion.chunk",
		Choices: []schemas.BifrostResponseChoice{{
			Index: 0,
			ChatStreamResponseChoice: &schemas.ChatStreamResponseChoice{
				Delta: &schemas.ChatStreamResponseChoiceDelta{
					ProviderBlocks: []schemas.ChatProviderBlock{{Index: open.shared, Provider: schemas.ProviderBlockAnthropic, Block: block}},
				},
			},
		}},
	}
}

// anthropicReasoningBlock maps one replayed reasoning detail to its Anthropic block.
func anthropicReasoningBlock(detail schemas.ChatReasoningDetails) AnthropicContentBlock {
	// reasoning.encrypted details carrying data hold an anthropic redacted_thinking
	// payload; replay the block as-is so the API can decrypt it. Encrypted details
	// without data (e.g. gemini thought signatures) keep the thinking-block mapping.
	if detail.Type == schemas.BifrostReasoningDetailsTypeEncrypted && detail.Data != nil && *detail.Data != "" {
		return AnthropicContentBlock{Type: AnthropicContentBlockTypeRedactedThinking, Data: detail.Data}
	}
	return AnthropicContentBlock{Type: AnthropicContentBlockTypeThinking, Signature: detail.Signature, Thinking: detail.Text}
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
		block, err := replayedProviderBlock(pb)
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
func replayedProviderBlock(pb schemas.ChatProviderBlock) (AnthropicContentBlock, error) {
	if pb.Provider != schemas.ProviderBlockAnthropic {
		return AnthropicContentBlock{}, fmt.Errorf("provider is not %q", schemas.ProviderBlockAnthropic)
	}
	raw := bytes.TrimSpace(pb.Block)
	if len(raw) == 0 || raw[0] != '{' {
		return AnthropicContentBlock{}, fmt.Errorf("block is not a JSON object")
	}
	block := AnthropicContentBlock{Raw: json.RawMessage(raw)}
	if t := gjson.GetBytes(raw, "type"); t.Type == gjson.String {
		block.Type = AnthropicContentBlockType(t.Str)
	}
	if cc := gjson.GetBytes(raw, "cache_control"); cc.Exists() {
		if cc.Type != gjson.Null {
			var c schemas.CacheControl
			if err := sonic.UnmarshalString(cc.Raw, &c); err != nil {
				return AnthropicContentBlock{}, fmt.Errorf("cache_control is not a cache control object")
			}
			block.CacheControl = &c
		}
		stripped, err := providerUtils.DeleteJSONField(raw, "cache_control")
		if err != nil {
			return AnthropicContentBlock{}, fmt.Errorf("cache_control could not be moved")
		}
		block.Raw = stripped
	}
	return block, nil
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
// verbatim, appending the typed cache_control as a replayed block's last member.
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
		end := bytes.LastIndexByte(b.Raw, '}')
		buf = append(buf, b.Raw[:end]...)
		if len(bytes.TrimSpace(b.Raw[1:end])) > 0 {
			buf = append(buf, ',')
		}
		buf = append(buf, `"cache_control":`...)
		buf = append(buf, cc...)
		buf = append(buf, '}')
	}
	return append(buf, ']'), nil
}

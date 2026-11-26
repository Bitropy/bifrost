package anthropic

import (
	"bytes"

	"github.com/maximhq/bifrost/core/schemas"
)

// Stream folding: an unmapped block is buffered from its content_block_start and emitted
// whole at its content_block_stop. The scratch holds the start object followed by the
// value its deltas accumulate, into one member of the block.
const (
	foldMemberInput   = "input"   // input_json_delta: partial_json fragments
	foldMemberContent = "content" // compaction_delta: escaped string fragments
)

// providerBlockScratch sits behind a pointer so growing it is not a store into the stream
// state itself, which escape analysis would read as the state's maps escaping.
type providerBlockScratch struct{ b []byte }

type openProviderBlock struct {
	open  bool
	index int
	// member is the block member the deltas fold into, set by the first delta; "" until then.
	member string
	objLen int
}

// providerBlockFold is the block-agnostic half of a stream's provider-block carry, shared by
// the chat and Responses stream states. Anthropic streams one content block at a time, so
// one open record is enough.
type providerBlockFold struct {
	openBlock openProviderBlock
	// scratch holds the open block's start object and its folded deltas. Allocated on the
	// stream's first unmapped block and reused for every later one.
	scratch            *providerBlockScratch
	providerBlockDrops schemas.ProviderBlockDrops
	// eventData is the current event's undecoded data, set by the stream loop before each
	// conversion; unmapped blocks are copied from it verbatim.
	eventData []byte
}

type foldState uint8

const (
	foldNone     foldState = iota // the event is not the fold's
	foldConsumed                  // the event belongs to the open block and yields nothing
	foldClosed                    // the open block closed; its bytes are returned
)

// interrupt drops a block still open when the next block starts. Anthropic does not produce
// one; the new block owns the event, so the open one is lost and counted. Every start calls
// it, before the wrapper decides whether to carry the new block.
func (f *providerBlockFold) interrupt() {
	if f.openBlock.open {
		f.openBlock = openProviderBlock{}
		f.providerBlockDrops.Interleaved++
	}
}

// event folds the delta and stop events of the open block. A content_block_start is never
// passed here: the wrapper decides on it and calls start.
func (f *providerBlockFold) event(chunk *AnthropicStreamEvent) ([]byte, foldState) {
	if chunk.Index == nil || !f.openBlock.open || f.openBlock.index != *chunk.Index {
		return nil, foldNone
	}
	switch chunk.Type {
	case AnthropicStreamEventTypeContentBlockDelta:
		if chunk.Delta != nil {
			f.fold(chunk.Delta)
			return nil, foldConsumed
		}
	case AnthropicStreamEventTypeContentBlockStop:
		return f.close(), foldClosed
	}
	return nil, foldNone
}

// start opens an unmapped block from the event's raw content_block. Without the event bytes
// there is nothing verbatim to copy, so the block is not tracked at all.
func (f *providerBlockFold) start(blockIndex int) bool {
	data := f.eventData
	s, e := jsonMember(data, 0, len(data), "content_block")
	if !isJSONObjectAt(data, s, e) {
		return false
	}
	f.openBlock = openProviderBlock{open: true, index: blockIndex, objLen: e - s}
	if f.scratch == nil {
		f.scratch = &providerBlockScratch{}
	}
	f.scratch.b = append(f.scratch.b[:0], data[s:e]...)
	return true
}

// fold appends one delta of the open block to the scratch. A delta it cannot fold, or one
// that would fold into a second member, is dropped and counted.
func (f *providerBlockFold) fold(delta *AnthropicStreamDelta) {
	var member string
	switch delta.Type {
	case AnthropicStreamDeltaTypeInputJSON:
		if delta.PartialJSON != nil {
			member = foldMemberInput
		}
	case AnthropicStreamDeltaTypeCompaction:
		member = foldMemberContent
	}
	// The compaction fragment is kept escaped as sent, so the fragments concatenate
	// into one JSON string.
	data := f.eventData
	ds, de := jsonMember(data, 0, len(data), "delta")
	cs, ce := jsonMember(data, ds, de, "content")
	if member == foldMemberContent && (cs < 0 || data[cs] != '"') {
		member = ""
	}
	if member == "" || (f.openBlock.member != "" && f.openBlock.member != member) {
		f.providerBlockDrops.UnknownDelta++
		return
	}
	if member == foldMemberContent && f.openBlock.member == "" {
		f.scratch.b = append(f.scratch.b, '"')
	}
	f.openBlock.member = member
	if member == foldMemberInput {
		f.scratch.b = append(f.scratch.b, *delta.PartialJSON...)
	} else {
		f.scratch.b = append(f.scratch.b, data[cs+1:ce-1]...)
	}
}

// close assembles the open block: the start object with the folded member set to the
// accumulated value. The splice is by hand, into one sized allocation, because sjson's path
// handling costs two more allocations per block.
func (f *providerBlockFold) close() []byte {
	open := f.openBlock
	f.openBlock = openProviderBlock{}
	obj, acc := f.scratch.b[:open.objLen], f.scratch.b[open.objLen:]
	member := open.member
	if member == foldMemberContent {
		acc = append(acc, '"')
	}

	if member == "" || len(acc) == 0 {
		return bytes.Clone(obj)
	}
	if vs, ve := jsonMember(obj, 0, len(obj), member); vs > 0 {
		block := make([]byte, 0, len(obj)-(ve-vs)+len(acc))
		return append(append(append(block, obj[:vs]...), acc...), obj[ve:]...)
	}
	end := bytes.LastIndexByte(obj, '}')
	block := make([]byte, 0, len(obj)+len(member)+4+len(acc))
	block = append(block, obj[:end]...)
	if len(bytes.TrimSpace(obj[1:end])) > 0 {
		block = append(block, ',')
	}
	block = append(append(append(append(block, '"'), member...), `":`...), acc...)
	return append(block, obj[end:]...)
}

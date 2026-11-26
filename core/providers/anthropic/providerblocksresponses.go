package anthropic

import (
	"encoding/json"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/tidwall/gjson"
)

// rawAnswerContent is the raw content array of the body handed over by SetRawBody, resolved
// on the first at call only, so an answer whose blocks all reach an item walks nothing.
type rawAnswerContent struct {
	body     []byte
	n        int
	arr      []gjson.Result
	resolved bool
}

// at returns the raw bytes of block i. ok is false without a body, or when the body's
// content array does not line up with the n decoded blocks. The bytes are a copy, so no
// item keeps the body alive.
func (r *rawAnswerContent) at(i int) (json.RawMessage, bool) {
	b, ok := r.result(i)
	if !ok {
		return nil, false
	}
	return json.RawMessage(b.Raw), true
}

// result is at as the resolved gjson value.
func (r *rawAnswerContent) result(i int) (gjson.Result, bool) {
	if r.body == nil {
		return gjson.Result{}, false
	}
	if !r.resolved {
		r.resolved = true
		r.arr = gjson.Get(bytesAsString(r.body), "content").Array()
	}
	if len(r.arr) != r.n || i < 0 || i >= r.n || !r.arr[i].IsObject() {
		return gjson.Result{}, false
	}
	return r.arr[i], true
}

// appendProviderBlockItem carries answer block i as a provider_block item at the current
// position. Without its raw bytes the block cannot be carried, so the loss is recorded.
func appendProviderBlockItem(ctx *schemas.BifrostContext, out []schemas.ResponsesMessage, raw *rawAnswerContent, i int) []schemas.ResponsesMessage {
	block, ok := raw.at(i)
	if !ok {
		schemas.AddAdaptation(ctx, schemas.AdaptationUnmapped)
		return out
	}
	return append(out, schemas.NewResponsesProviderBlock(newProviderBlockID(), schemas.ProviderBlockAnthropic, block))
}

// isWholeEmitted reports whether the stream emits a block of type t whole at its
// content_block_start, so its deltas and stop carry nothing.
func isWholeEmitted(t AnthropicContentBlockType) bool {
	switch t {
	case AnthropicContentBlockTypeImage, AnthropicContentBlockTypeDocument, AnthropicContentBlockTypeContainerUpload,
		AnthropicContentBlockTypeToolResult, AnthropicContentBlockTypeMCPToolResult,
		AnthropicContentBlockTypeWebSearchToolResultError, AnthropicContentBlockTypeFallback:
		return true
	}
	return false
}

// openCallMatches reports whether a result's tool_use_id names the call open on the stream.
func openCallMatches(open, toolUseID *string) bool {
	return open != nil && toolUseID != nil && *open == *toolUseID
}

// toolUsePlaced reports whether a tool_use block has an arm: the structured-output tool, or a
// call with an id and a name. Unary and stream both decide with it.
func toolUsePlaced(b *AnthropicContentBlock, structuredOutputToolName string) bool {
	if structuredOutputToolName != "" && b.Name != nil && *b.Name == structuredOutputToolName {
		return true
	}
	return b.ID != nil && b.Name != nil
}

// serverToolUsePlaced reports whether a server_tool_use block has an arm: an id and a name
// with a Responses arm. Unary and stream both decide with it.
func serverToolUsePlaced(b *AnthropicContentBlock) bool {
	return b.ID != nil && b.Name != nil && isResponsesServerToolName(*b.Name)
}

// isResponsesServerToolName reports whether a server_tool_use of this name has a Responses
// arm; any other server tool is carried as a provider_block item.
func isResponsesServerToolName(name string) bool {
	switch name {
	case string(AnthropicToolNameWebSearch), string(AnthropicToolNameWebFetch), string(AnthropicToolNameAdvisor):
		return true
	}
	return isAnthropicToolSearchToolName(name) || isAnthropicCodeExecutionToolName(name)
}

// providerBlockEvent hands the deltas and stop of an open provider block to the fold, ahead
// of the typed conversion. It reports whether it handled the event. Starts are decided by the typed arms (carryUnplacedStart).
func (state *AnthropicResponsesStreamState) providerBlockEvent(chunk *AnthropicStreamEvent, sequenceNumber int) ([]*schemas.BifrostResponsesStreamResponse, bool) {
	block, st := state.event(chunk)
	switch st {
	case foldConsumed:
		return nil, true
	case foldClosed:
		delete(state.ContentIndexToBlockType, *chunk.Index)
		item := schemas.NewResponsesProviderBlock(state.pbID, schemas.ProviderBlockAnthropic, block)
		return state.emitWholeItem(state.pbOutputIndex, chunk.Index, &item, sequenceNumber), true
	}
	return nil, false
}

// carryUnplacedStart opens a provider block for a content_block_start no typed arm placed.
// Without the event bytes there is nothing verbatim to carry, so the loss is recorded.
func (state *AnthropicResponsesStreamState) carryUnplacedStart(ctx *schemas.BifrostContext, chunk *AnthropicStreamEvent) {
	if !state.start(*chunk.Index) {
		schemas.AddAdaptation(ctx, schemas.AdaptationUnmapped)
		return
	}
	state.pbOutputIndex = state.getOrCreateOutputIndex(chunk.Index)
	state.pbID = newProviderBlockID()
	state.ContentIndexToBlockType[*chunk.Index] = chunk.ContentBlock.Type
}

// emitWholeItem keeps a complete item for response.completed and returns its
// output_item.added and output_item.done.
func (state *AnthropicResponsesStreamState) emitWholeItem(outputIndex int, contentIndex *int, item *schemas.ResponsesMessage, sequenceNumber int) []*schemas.BifrostResponsesStreamResponse {
	stored := *item
	state.OutputItems[outputIndex] = &stored
	return []*schemas.BifrostResponsesStreamResponse{
		{
			Type:           schemas.ResponsesStreamResponseTypeOutputItemAdded,
			SequenceNumber: sequenceNumber,
			OutputIndex:    schemas.Ptr(outputIndex),
			ContentIndex:   contentIndex,
			Item:           item,
		},
		{
			Type:           schemas.ResponsesStreamResponseTypeOutputItemDone,
			SequenceNumber: sequenceNumber + 1,
			OutputIndex:    schemas.Ptr(outputIndex),
			ContentIndex:   contentIndex,
			Item:           item,
		},
	}
}

// replayedResponsesProviderBlock is the content block of a replayed provider_block item. An
// item core cannot carry is left out of the turn and the loss recorded: the converter has no
// error result, and the router refuses the same items before dispatch.
func replayedResponsesProviderBlock(ctx *schemas.BifrostContext, msg *schemas.ResponsesMessage) (AnthropicContentBlock, bool) {
	provider, raw, ok := msg.ProviderBlock()
	if ok {
		block, err := replayedProviderBlock(provider, raw)
		if err == nil {
			return block, true
		}
	}
	schemas.AddAdaptation(ctx, schemas.AdaptationUnmapped)
	return AnthropicContentBlock{}, false
}

// appendJoiningReplayedTurn appends msg, folding an assistant message into the assistant turn
// before it when that turn ends with a replayed block: the items around a replayed block
// came from one Anthropic turn.
func appendJoiningReplayedTurn(msgs []AnthropicMessage, msg AnthropicMessage) []AnthropicMessage {
	last := len(msgs) - 1
	if last < 0 || msg.Role != AnthropicMessageRoleAssistant || msgs[last].Role != AnthropicMessageRoleAssistant ||
		msg.Content.ContentStr != nil || msgs[last].Content.ContentStr != nil {
		return append(msgs, msg)
	}
	prev := msgs[last].Content.ContentBlocks
	if len(prev) == 0 || prev[len(prev)-1].Raw == nil {
		return append(msgs, msg)
	}
	msgs[last].Content.ContentBlocks = append(prev, msg.Content.ContentBlocks...)
	return msgs
}

// isPayloadFreeBlock reports whether a block's only content is its type: a redacted_thinking
// without data or a compaction without content or tool_changes. Omitting one loses nothing, so it counts as
// placed in both modes.
func isPayloadFreeBlock(b *AnthropicContentBlock) bool {
	switch b.Type {
	case AnthropicContentBlockTypeRedactedThinking:
		return b.Data == nil || *b.Data == ""
	case AnthropicContentBlockTypeCompaction:
		return b.Content == nil && len(b.ToolChanges) == 0
	}
	return false
}

// newProviderBlockID is the id of a provider_block item, the same on unary and stream.
func newProviderBlockID() string { return "pb_" + schemas.GetRandomString(50) }

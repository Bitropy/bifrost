package schemas

// ChatMessagesWithoutCacheControl returns msgs with every cache_control off
// their content blocks, tool calls and tool results, for a chat wire that has no cache
// breakpoint and serialises the messages as they are. Only the messages it
// changes are copied: msgs is the caller's, reused by fallbacks. A
// TextJoinOnWire content already marshals without its markers.
func ChatMessagesWithoutCacheControl(msgs []ChatMessage) []ChatMessage {
	var out []ChatMessage
	for i := range msgs {
		m := msgs[i]
		changed := false
		if c := m.Content; c != nil && !c.TextJoinOnWire && blocksMarked(c.ContentBlocks) {
			cc := *c
			cc.ContentBlocks = append([]ChatContentBlock(nil), c.ContentBlocks...)
			for j := range cc.ContentBlocks {
				cc.ContentBlocks[j].CacheControl = nil
			}
			m.Content, changed = &cc, true
		}
		if a := m.ChatAssistantMessage; a != nil && toolCallsMarked(a.ToolCalls) {
			ac := *a
			ac.ToolCalls = append([]ChatAssistantMessageToolCall(nil), a.ToolCalls...)
			for j := range ac.ToolCalls {
				ac.ToolCalls[j].CacheControl = nil
			}
			m.ChatAssistantMessage, changed = &ac, true
		}
		if t := m.ChatToolMessage; t != nil && t.CacheControl != nil {
			tc := *t
			tc.CacheControl = nil
			m.ChatToolMessage, changed = &tc, true
		}
		if !changed {
			continue
		}
		if out == nil {
			out = append([]ChatMessage(nil), msgs...)
		}
		out[i] = m
	}
	if out == nil {
		return msgs
	}
	return out
}

// ChatToolsWithoutCacheControl is tools without their cache_control, copied
// only when one carries it.
func ChatToolsWithoutCacheControl(tools []ChatTool) []ChatTool {
	for i := range tools {
		if tools[i].CacheControl == nil {
			continue
		}
		out := append([]ChatTool(nil), tools...)
		for j := range out {
			out[j].CacheControl = nil
		}
		return out
	}
	return tools
}

func blocksMarked(blocks []ChatContentBlock) bool {
	for i := range blocks {
		if blocks[i].CacheControl != nil {
			return true
		}
	}
	return false
}

func toolCallsMarked(calls []ChatAssistantMessageToolCall) bool {
	for i := range calls {
		if calls[i].CacheControl != nil {
			return true
		}
	}
	return false
}

// ReportChatCacheControlDropped records AdaptationUnmapped on ctx when req
// carries a cache_control marker, for a converter whose wire has no place for
// one: the client's caching intent is lost, so the attempt says so.
func ReportChatCacheControlDropped(ctx *BifrostContext, req *BifrostChatRequest) {
	if req == nil {
		return
	}
	var tools []ChatTool
	if req.Params != nil {
		tools = req.Params.Tools
	}
	if chatMessagesMarked(req.Input, false) || chatToolsMarked(tools) {
		AddAdaptation(ctx, AdaptationUnmapped)
	}
}

// chatMessagesMarked reports a cache_control on msgs' blocks, tool calls or
// tool results; skipToolResults leaves out a tool message, whose markers a
// function_call_output carries.
func chatMessagesMarked(msgs []ChatMessage, skipToolResults bool) bool {
	for i := range msgs {
		m := &msgs[i]
		if skipToolResults && m.Role == ChatMessageRoleTool {
			continue
		}
		if m.ChatToolMessage != nil && m.ChatToolMessage.CacheControl != nil {
			return true
		}
		if m.Content != nil && blocksMarked(m.Content.ContentBlocks) {
			return true
		}
		if m.ChatAssistantMessage != nil && toolCallsMarked(m.ChatAssistantMessage.ToolCalls) {
			return true
		}
	}
	return false
}

func chatToolsMarked(tools []ChatTool) bool {
	for i := range tools {
		if tools[i].CacheControl != nil {
			return true
		}
	}
	return false
}

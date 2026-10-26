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

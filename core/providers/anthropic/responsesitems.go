package anthropic

// Responses items built from Anthropic answer blocks, shared by the unary and stream
// converters, and the helpers that fold a result block onto its call.

import "github.com/maximhq/bifrost/core/schemas"

// anthropicContentPartItem is the message item of one image, document or container_upload
// content part.
func anthropicContentPartItem(part schemas.ResponsesMessageContentBlock, role *schemas.ResponsesMessageRoleType, isOutputMessage bool) schemas.ResponsesMessage {
	item := schemas.ResponsesMessage{
		Type:    schemas.Ptr(schemas.ResponsesMessageTypeMessage),
		Role:    role,
		Content: &schemas.ResponsesMessageContent{ContentBlocks: []schemas.ResponsesMessageContentBlock{part}},
	}
	if isOutputMessage {
		item.ID = schemas.Ptr(newMessageItemID())
		item.Status = schemas.Ptr("completed")
	}
	return item
}

// anthropicFallbackItem is the message item of a server-side fallback boundary block.
func anthropicFallbackItem(block *AnthropicContentBlock, id string, role *schemas.ResponsesMessageRoleType) schemas.ResponsesMessage {
	fallback := &schemas.ResponsesOutputMessageContentFallback{}
	if block.From != nil {
		fallback.FromModel = block.From.Model
	}
	if block.To != nil {
		fallback.ToModel = block.To.Model
	}
	if block.Trigger != nil {
		fallback.TriggerType = block.Trigger.Type
		fallback.TriggerCategory = block.Trigger.Category
	}
	return schemas.ResponsesMessage{
		ID:     &id,
		Type:   schemas.Ptr(schemas.ResponsesMessageTypeMessage),
		Role:   role,
		Status: schemas.Ptr("completed"),
		Content: &schemas.ResponsesMessageContent{
			ContentBlocks: []schemas.ResponsesMessageContentBlock{{
				Type:                                  schemas.ResponsesOutputMessageContentTypeFallback,
				ResponsesOutputMessageContentFallback: fallback,
			}},
		},
	}
}

// anthropicToolResultItem is the function_call_output item of a tool_result block with a
// tool_use_id.
func anthropicToolResultItem(block *AnthropicContentBlock, isOutputMessage bool) schemas.ResponsesMessage {
	bifrostMsg := schemas.ResponsesMessage{
		Type:         schemas.Ptr(schemas.ResponsesMessageTypeFunctionCallOutput),
		Status:       schemas.Ptr("completed"),
		CacheControl: block.CacheControl,
		ResponsesToolMessage: &schemas.ResponsesToolMessage{
			CallID:      block.ToolUseID,
			ToolsetName: block.ToolsetName,
		},
	}
	// Initialize the nested struct before any writes
	bifrostMsg.Output = &schemas.ResponsesToolMessageOutputStruct{}

	if block.Content != nil && block.Content.ContentStr != nil {
		bifrostMsg.Output.ResponsesToolCallOutputStr = block.Content.ContentStr
	} else if block.Content != nil && block.Content.ContentBlocks != nil {
		var toolMsgContentBlocks []schemas.ResponsesMessageContentBlock
		for _, contentBlock := range block.Content.ContentBlocks {
			switch contentBlock.Type {
			case AnthropicContentBlockTypeText:
				if contentBlock.Text != nil {
					toolMsgContentBlocks = append(toolMsgContentBlocks, toolResultTextBlock(&contentBlock, isOutputMessage))
				}
			case AnthropicContentBlockTypeImage:
				if contentBlock.Source != nil && contentBlock.Source.SourceObj != nil {
					toolMsgContentBlocks = append(toolMsgContentBlocks, contentBlock.toBifrostResponsesImageBlock())
				}
			case AnthropicContentBlockTypeDocument:
				// A tool returning a PDF (Claude Code's Read tool, for one)
				// nests a document block here. Dropping it leaves the
				// function_call_output with an empty content array, which
				// OpenAI rejects outright:
				// "Missing required parameter: 'input[N].content[0]'".
				if contentBlock.Source != nil && contentBlock.Source.SourceObj != nil {
					toolMsgContentBlocks = append(toolMsgContentBlocks, contentBlock.toBifrostResponsesDocumentBlock())
				}
			case AnthropicContentBlockTypeToolReference:
				if contentBlock.ToolName != nil {
					toolMsgContentBlocks = append(toolMsgContentBlocks, schemas.ResponsesMessageContentBlock{
						Type:     schemas.ResponsesInputMessageContentBlockTypeToolReference,
						ToolName: contentBlock.ToolName,
					})
				}
			}
		}
		bifrostMsg.Output.ResponsesFunctionToolCallOutputBlocks = toolMsgContentBlocks
	}
	// Handle is_error from Anthropic
	if block.IsError != nil && *block.IsError {
		bifrostMsg.Status = schemas.Ptr("incomplete")
	}
	return bifrostMsg
}

// anthropicMCPToolResultItem is the mcp_call item of an mcp_tool_result block with a
// tool_use_id.
func anthropicMCPToolResultItem(block *AnthropicContentBlock, isOutputMessage bool) schemas.ResponsesMessage {
	bifrostMsg := schemas.ResponsesMessage{
		Type:   schemas.Ptr(schemas.ResponsesMessageTypeMCPCall),
		Status: schemas.Ptr("completed"),
		ResponsesToolMessage: &schemas.ResponsesToolMessage{
			CallID: block.ToolUseID,
		},
	}
	if isOutputMessage {
		bifrostMsg.ID = schemas.Ptr(newMessageItemID())
	}
	// Initialize the nested struct before any writes
	bifrostMsg.Output = &schemas.ResponsesToolMessageOutputStruct{}

	if block.Content != nil {
		if block.Content.ContentStr != nil {
			bifrostMsg.Output.ResponsesToolCallOutputStr = block.Content.ContentStr
		} else if block.Content.ContentBlocks != nil {
			var toolMsgContentBlocks []schemas.ResponsesMessageContentBlock
			for _, contentBlock := range block.Content.ContentBlocks {
				if contentBlock.Type == AnthropicContentBlockTypeText {
					if contentBlock.Text != nil {
						toolMsgContentBlocks = append(toolMsgContentBlocks, toolResultTextBlock(&contentBlock, isOutputMessage))
					}
				}
			}
			bifrostMsg.Output.ResponsesFunctionToolCallOutputBlocks = toolMsgContentBlocks
		}
	}
	return bifrostMsg
}

// buildWholeItem is the item of an image, document, container_upload, tool_result or
// mcp_tool_result block, behind the block's guard; ok is false when the block lacks what
// its item needs. Unary and stream call it, so both place the same blocks.
func buildWholeItem(block *AnthropicContentBlock, role *schemas.ResponsesMessageRoleType, isOutputMessage bool) (schemas.ResponsesMessage, bool) {
	switch block.Type {
	case AnthropicContentBlockTypeImage, AnthropicContentBlockTypeDocument:
		if block.Source == nil || block.Source.SourceObj == nil {
			return schemas.ResponsesMessage{}, false
		}
		if block.Type == AnthropicContentBlockTypeImage {
			return anthropicContentPartItem(block.toBifrostResponsesImageBlock(), role, isOutputMessage), true
		}
		return anthropicContentPartItem(block.toBifrostResponsesDocumentBlock(), role, isOutputMessage), true
	case AnthropicContentBlockTypeContainerUpload:
		if block.FileID == nil {
			return schemas.ResponsesMessage{}, false
		}
		return anthropicContentPartItem(block.toBifrostResponsesContainerUploadBlock(), role, isOutputMessage), true
	case AnthropicContentBlockTypeToolResult, AnthropicContentBlockTypeMCPToolResult:
		if block.ToolUseID == nil {
			return schemas.ResponsesMessage{}, false
		}
		if block.Type == AnthropicContentBlockTypeToolResult {
			return anthropicToolResultItem(block, isOutputMessage), true
		}
		return anthropicMCPToolResultItem(block, isOutputMessage), true
	}
	return schemas.ResponsesMessage{}, false
}

// compactionStreamItem is a streamed compaction's item, with the cache control and
// tool_changes its start recorded.
func (state *AnthropicResponsesStreamState) compactionStreamItem(outputIndex int, summary string) *schemas.ResponsesMessage {
	itemID := state.ItemIDs[outputIndex]
	return &schemas.ResponsesMessage{
		ID:     &itemID,
		Status: schemas.Ptr("completed"),
		Type:   schemas.Ptr(schemas.ResponsesMessageTypeMessage),
		Role:   schemas.Ptr(schemas.ResponsesInputMessageRoleAssistant),
		Content: &schemas.ResponsesMessageContent{
			ContentBlocks: []schemas.ResponsesMessageContentBlock{{
				Type: schemas.ResponsesOutputMessageContentTypeCompaction,
				ResponsesOutputMessageContentCompaction: &schemas.ResponsesOutputMessageContentCompaction{
					Summary:     summary,
					ToolChanges: state.CompactionToolChanges[outputIndex],
				},
				CacheControl: state.CompactionContentIndices[outputIndex],
			}},
		},
	}
}

// webSearchResultPlaced reports whether a web_search_tool_result answers the open search,
// or, with none open, one deferred in the previous assistant message. A result for another
// call while a search is open is not placed. Unary and stream both decide with it.
func webSearchResultPlaced(open, toolUseID *string) bool {
	return toolUseID != nil && (open == nil || *open == *toolUseID)
}

// openWebSearchCallID is the unary counterpart of the stream's open search: the latest
// web_search_call still waiting for its result.
func openWebSearchCallID(msgs []schemas.ResponsesMessage) *string {
	for i := len(msgs) - 1; i >= 0; i-- {
		m := &msgs[i]
		if m.Type != nil && *m.Type == schemas.ResponsesMessageTypeWebSearchCall && m.Status != nil && *m.Status == "in_progress" {
			return m.ID
		}
	}
	return nil
}

// attachAdvisorResult folds an advisor_tool_result onto its advisor_call and reports whether
// one took it.
func attachAdvisorResult(msgs []schemas.ResponsesMessage, toolUseID string, block *AnthropicContentBlock) bool {
	for i := len(msgs) - 1; i >= 0; i-- {
		msg := &msgs[i]
		if msg.Type == nil || *msg.Type != schemas.ResponsesMessageTypeAdvisorCall ||
			msg.ResponsesToolMessage == nil || msg.CallID == nil || *msg.CallID != toolUseID {
			continue
		}
		advisor := &schemas.ResponsesAdvisorCall{}
		// content is a single object on the wire (ContentObj); UnmarshalJSON wraps incoming
		// objects into ContentBlocks, so accept either.
		if c := block.Content; c != nil {
			inner := c.ContentObj
			if inner == nil && len(c.ContentBlocks) > 0 {
				inner = &c.ContentBlocks[0]
			}
			if inner != nil {
				advisor.ResultType = string(inner.Type)
				advisor.Text = inner.Text
				advisor.EncryptedContent = inner.EncryptedContent
				advisor.ErrorCode = inner.ErrorCode
				advisor.StopReason = inner.StopReason
			}
		}
		msg.ResponsesAdvisorCall = advisor
		return true
	}
	return false
}

// markWebSearchFailed marks the web_search_call of toolUseID failed and reports whether one
// was there.
func markWebSearchFailed(msgs []schemas.ResponsesMessage, toolUseID string) bool {
	for i := len(msgs) - 1; i >= 0; i-- {
		msg := &msgs[i]
		if msg.Type != nil && *msg.Type == schemas.ResponsesMessageTypeWebSearchCall && msg.ID != nil && *msg.ID == toolUseID {
			msg.Status = schemas.Ptr("failed")
			return true
		}
	}
	return false
}

// toolResultTextBlock is a text block inside a tool result, typed for the message's side.
func toolResultTextBlock(block *AnthropicContentBlock, isOutputMessage bool) schemas.ResponsesMessageContentBlock {
	blockType := schemas.ResponsesInputMessageContentBlockTypeText
	if isOutputMessage {
		blockType = schemas.ResponsesOutputMessageContentTypeText
	}
	return schemas.ResponsesMessageContentBlock{Type: blockType, Text: block.Text, CacheControl: block.CacheControl}
}

// newMessageItemID is the id an output item built here gets.
func newMessageItemID() string { return "msg_" + schemas.GetRandomString(50) }

package anthropic

import (
	"context"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

// Stripping cache_control.scope for a provider without prompt-caching scope
// (Vertex) clears it on the converted request only: the caller's request
// shares its *CacheControl pointers with the conversion (top level, each
// tool, each content block), and every attempt of a failover chain converts
// the same request, so a write through them cost a later scope-capable
// attempt its scope and raced an abandoned worker against the next attempt
// (BIT-2595, BIT-2741).
func TestToAnthropicChatRequest_ScopeStripCopiesBeforeWrite(t *testing.T) {
	scope := "user"
	cc := func() *schemas.CacheControl {
		return &schemas.CacheControl{Type: schemas.CacheControlTypeEphemeral, Scope: &scope}
	}
	req := &schemas.BifrostChatRequest{
		Provider: schemas.Vertex,
		Model:    "claude-sonnet-4-5",
		Input: []schemas.ChatMessage{
			{Role: schemas.ChatMessageRoleSystem, Content: &schemas.ChatMessageContent{ContentBlocks: []schemas.ChatContentBlock{
				{Type: schemas.ChatContentBlockTypeText, Text: schemas.Ptr("sys"), CacheControl: cc()},
			}}},
			{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentBlocks: []schemas.ChatContentBlock{
				{Type: schemas.ChatContentBlockTypeText, Text: schemas.Ptr("hi"), CacheControl: cc()},
			}}},
		},
		Params: &schemas.ChatParameters{
			CacheControl: cc(),
			Tools: []schemas.ChatTool{{
				Type:         schemas.ChatToolTypeFunction,
				Function:     &schemas.ChatToolFunction{Name: "f"},
				CacheControl: cc(),
			}},
		},
	}
	out, err := ToAnthropicChatRequest(schemas.NewBifrostContext(context.Background(), schemas.NoDeadline), req)
	if err != nil {
		t.Fatal(err)
	}

	if out.CacheControl != nil && out.CacheControl.Scope != nil {
		t.Error("the converted request keeps the top-level scope on Vertex")
	}
	for _, tool := range out.Tools {
		if tool.CacheControl != nil && tool.CacheControl.Scope != nil {
			t.Error("the converted request keeps a tool scope on Vertex")
		}
	}
	for _, m := range out.Messages {
		for _, b := range m.Content.ContentBlocks {
			if b.CacheControl != nil && b.CacheControl.Scope != nil {
				t.Error("the converted request keeps a content-block scope on Vertex")
			}
		}
	}
	if out.System != nil {
		for _, b := range out.System.ContentBlocks {
			if b.CacheControl != nil && b.CacheControl.Scope != nil {
				t.Error("the converted request keeps a system-block scope on Vertex")
			}
		}
	}

	for name, c := range map[string]*schemas.CacheControl{
		"params":        req.Params.CacheControl,
		"tool":          req.Params.Tools[0].CacheControl,
		"system block":  req.Input[0].Content.ContentBlocks[0].CacheControl,
		"message block": req.Input[1].Content.ContentBlocks[0].CacheControl,
	} {
		if c == nil || c.Scope == nil || *c.Scope != scope {
			t.Errorf("the caller's %s cache_control lost its scope: %+v", name, c)
		}
	}
}

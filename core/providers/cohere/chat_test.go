package cohere

import (
	"strings"
	"testing"

	"github.com/maximhq/bifrost/core/internal/wirecapture"
	"github.com/maximhq/bifrost/core/schemas"
)

// Cohere tool messages take text: media a tool returned becomes a placeholder naming it.
func TestToCohereChatCompletionRequest_ToolResultNamesMedia(t *testing.T) {
	req, err := ToCohereChatCompletionRequest(nil, &schemas.BifrostChatRequest{
		Provider: schemas.Cohere,
		Model:    "command-a-03-2025",
		Input: []schemas.ChatMessage{
			{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("plot it")}},
			{Role: schemas.ChatMessageRoleTool, ChatToolMessage: &schemas.ChatToolMessage{ToolCallID: schemas.Ptr("c1")}, Content: &schemas.ChatMessageContent{ContentBlocks: []schemas.ChatContentBlock{
		{Type: schemas.ChatContentBlockTypeText, Text: schemas.Ptr("chart below")},
		{Type: schemas.ChatContentBlockTypeImage, ImageURLStruct: &schemas.ChatInputImage{URL: "data:image/png;base64,iVBORw0KGgo="}},
		{Type: schemas.ChatContentBlockTypeInputAudio, InputAudio: &schemas.ChatInputAudio{Data: "SUQz", Format: schemas.Ptr("mp3")}},
	}}},
		},
	})
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	wire, err := schemas.MarshalSorted(req.Messages[1])
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	for _, want := range []string{"chart below", "[image/png", "[audio/mp3"} {
		if !strings.Contains(string(wire), want) {
			t.Fatalf("expected %q in %s", want, wire)
		}
	}
	if strings.Contains(string(wire), "image_url") {
		t.Fatalf("no image part may reach a Cohere tool message: %s", wire)
	}
}

// BIT-2906: a tool-result image reaches Cohere's text-only tool message as a placeholder: without a size under a
// request-body rewriter, upstream's sized text without one. Unary and streaming alike.
func TestCohereChat_ToolResultMediaPlaceholderOmitsSizeUnderRewriter(t *testing.T) {
	server := wirecapture.NewServer(t)
	config := &schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{BaseURL: server.URL, DefaultRequestTimeoutInSeconds: 30}}
	provider, err := NewCohereProvider(config, wirecapture.Logger{})
	if err != nil {
		t.Fatal(err)
	}
	key := schemas.Key{Value: *schemas.NewSecretVar("test-key")}
	request := func() *schemas.BifrostChatRequest {
		return &schemas.BifrostChatRequest{Provider: schemas.Cohere, Model: "command-a-03-2025", Input: wirecapture.ChatToolResultImage(wirecapture.PNGDataURL)}
	}
	wirecapture.AssertChatPlaceholderSizeFreeUnderRewriter(t, server, provider, key, request)
}

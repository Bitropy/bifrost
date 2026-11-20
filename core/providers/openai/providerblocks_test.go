package openai

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/require"
)

// provider_blocks is read on the way in and never written on the OpenAI wire:
// the decode copies it onto the Bifrost message, the outbound conversion drops it.
func TestProviderBlocks_DecodedInDroppedOnWire(t *testing.T) {
	body := `{"model":"m","messages":[{"role":"user","content":"hi"},` +
		`{"role":"assistant","provider_blocks":[{"index":0,"provider":"anthropic","block":{"type":"server_tool_use","id":"s","name":"web_search","input":{}}}]}]}`
	var req OpenAIChatRequest
	require.NoError(t, json.Unmarshal([]byte(body), &req))
	msgs := ConvertOpenAIMessagesToBifrostMessages(req.Messages)
	require.NotNil(t, msgs[1].ChatAssistantMessage)
	require.Len(t, msgs[1].ChatAssistantMessage.ProviderBlocks, 1)
	require.JSONEq(t, `{"type":"server_tool_use","id":"s","name":"web_search","input":{}}`, string(msgs[1].ChatAssistantMessage.ProviderBlocks[0].Block))

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	wire, err := json.Marshal(&OpenAIChatRequest{Model: "gpt-4o", Messages: ConvertBifrostMessagesToOpenAIMessages(ctx, msgs)})
	require.NoError(t, err)
	require.False(t, strings.Contains(string(wire), "provider_blocks"), "provider_blocks reached the OpenAI wire: %s", wire)
}

// provider_citation annotations are Bifrost-only: the copy-on-strip drops them on
// the OpenAI wire, and every url_citation keeps its bytes.
func TestOpenAIChatRequest_DropsProviderCitations(t *testing.T) {
	start, end := 0, 5
	urlAnn := schemas.ChatAssistantMessageAnnotation{Type: "url_citation", URLCitation: schemas.ChatAssistantMessageAnnotationCitation{
		StartIndex: 0, EndIndex: 5, Title: "t", URL: schemas.Ptr("https://a"),
	}}
	req := &OpenAIChatRequest{Model: "gpt-4o", Provider: schemas.OpenAI, Messages: []OpenAIMessage{{
		Role:    schemas.ChatMessageRoleAssistant,
		Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("Alpha")},
		OpenAIChatAssistantMessage: &OpenAIChatAssistantMessage{Annotations: []schemas.ChatAssistantMessageAnnotation{
			{Type: schemas.ChatAnnotationTypeProviderCitation, Provider: "anthropic", StartIndex: &start, EndIndex: &end, Citation: json.RawMessage(`{"type":"char_location"}`)},
			urlAnn,
		}},
	}}}
	wire, err := json.Marshal(req)
	require.NoError(t, err)
	require.NotContains(t, string(wire), "provider_citation")
	require.NotContains(t, string(wire), "char_location")
	urlOnly, err := json.Marshal(urlAnn)
	require.NoError(t, err)
	require.Contains(t, string(wire), `"annotations":[`+string(urlOnly)+`]`)
	require.Len(t, req.Messages[0].OpenAIChatAssistantMessage.Annotations, 2, "the caller's message must not be mutated")
}

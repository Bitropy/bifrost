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

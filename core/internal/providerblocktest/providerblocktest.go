// Package providerblocktest holds the provider_block fixture shared by the tests of every
// provider that converts Claude answers through the anthropic package.
package providerblocktest

import (
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/tidwall/gjson"
)

// FutureBlock is a content block of a type core does not know, and FutureBlockAnswer an
// answer carrying it between two text blocks.
const (
	FutureBlock       = `{"type":"future_block","x":[1,"a"]}`
	FutureBlockAnswer = `{"id":"msg_1","type":"message","role":"assistant","model":"claude-x","stop_reason":"end_turn",` +
		`"content":[{"type":"text","text":"a"},` + FutureBlock + `,{"type":"text","text":"b"}],"usage":{"input_tokens":3,"output_tokens":4}}`
)

// CarriedProviderBlocks returns the block of every provider_block item in output, as sent.
func CarriedProviderBlocks(tb testing.TB, output []schemas.ResponsesMessage) []string {
	tb.Helper()
	var blocks []string
	for i := range output {
		if output[i].Type != nil && *output[i].Type == schemas.ResponsesMessageTypeProviderBlock {
			b, err := output[i].MarshalJSON()
			if err != nil {
				tb.Fatalf("marshal provider_block item %d: %v", i, err)
			}
			blocks = append(blocks, gjson.GetBytes(b, "block").Raw)
		}
	}
	return blocks
}

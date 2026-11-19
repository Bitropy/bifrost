package openai

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOpenAIResponsesRequestUnknownFields(t *testing.T) {
	body := []byte(`{"model":"m","input":"hi","Temperature":0.5,"max_output_tokens":8,"stream":true,"fallbacks":["x"],"extra_params":{},` +
		`"inference_geo":"us","mcp_servers":[ {"name": "m"} ],"thinking":{"type":"adaptive"}}`)
	got := OpenAIResponsesRequestUnknownFields(body)
	require.Len(t, got, 3)
	assert.Equal(t, json.RawMessage(`"us"`), got["inference_geo"])
	assert.Equal(t, json.RawMessage(`[{"name":"m"}]`), got["mcp_servers"], "compacted")
	assert.Equal(t, json.RawMessage(`{"type":"adaptive"}`), got["thinking"])

	// Decoding is unchanged: the type itself keeps no unknown key.
	var req OpenAIResponsesRequest
	require.NoError(t, json.Unmarshal(body, &req))
	assert.Nil(t, req.ExtraParams)
	assert.Nil(t, req.ResponsesParameters.ExtraParams)

	assert.Nil(t, OpenAIResponsesRequestUnknownFields(nil))
	known := []byte(`{"model":"m","input":"hi","temperature":0.5,"reasoning":{"effort":"high"},"tools":[]}`)
	assert.Nil(t, OpenAIResponsesRequestUnknownFields(known))
	allocs := testing.AllocsPerRun(100, func() { _ = OpenAIResponsesRequestUnknownFields(known) })
	assert.Zero(t, allocs, "a body with only known fields allocates nothing")
}

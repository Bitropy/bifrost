package openai

import (
	"context"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestSamplingStrip_EveryBuilder runs one row table through the chat builder,
// the Responses builder and ResponsesSamplingStrip: which sampling fields reach
// the wire, and whether the attempt records AdaptationSampling.
func TestSamplingStrip_EveryBuilder(t *testing.T) {
	no := false
	rows := []struct {
		name     string
		provider schemas.ModelProvider
		model    string
		effort   string
		record   *schemas.ModelCapabilities
		unset    bool // the request carries no sampling field
		stripped bool
	}{
		{name: "reasoning model, omitted effort", provider: schemas.OpenAI, model: "gpt-5-mini", stripped: true},
		{name: "reasoning model on azure", provider: schemas.Azure, model: "gpt-5-mini", effort: "high", stripped: true},
		{name: "gpt-5.4 effort none keeps sampling", provider: schemas.OpenAI, model: "gpt-5.4", effort: "none"},
		{name: "non-reasoning model keeps sampling", provider: schemas.OpenAI, model: "gpt-4.1"},
		{name: "supports_sampling false on any name", provider: schemas.Azure, model: "house-model", record: &schemas.ModelCapabilities{SupportsSamplingParams: &no}, stripped: true},
		{name: "supports_sampling false at effort none", provider: schemas.OpenAI, model: "gpt-5.4", effort: "none", record: &schemas.ModelCapabilities{SupportsSamplingParams: &no}, stripped: true},
		{
			name: "explicit unsupported_fields false outranks the fact", provider: schemas.OpenAI, model: "house-model",
			record: &schemas.ModelCapabilities{SupportsSamplingParams: &no, UnsupportedFields: map[string]bool{schemas.FieldTemperature: false, schemas.FieldTopP: false}},
		},
		{name: "nothing to strip records nothing", provider: schemas.OpenAI, model: "gpt-5-mini", unset: true},
	}
	for _, row := range rows {
		t.Run(row.name, func(t *testing.T) {
			if row.record != nil {
				installCapabilityRecord(t, row.model, row.record)
			}
			var temp, topP *float64
			if !row.unset {
				temp, topP = schemas.Ptr(0.7), schemas.Ptr(0.9)
			}
			var effort *string
			if row.effort != "" {
				effort = schemas.Ptr(row.effort)
			}
			user := schemas.Ptr("hi")

			ctx := samplingCtx(t)
			chat := ToOpenAIChatRequest(ctx, &schemas.BifrostChatRequest{
				Provider: row.provider, Model: row.model,
				Input:  []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: user}}},
				Params: &schemas.ChatParameters{Temperature: temp, TopP: topP, Reasoning: chatReasoning(effort)},
			})
			require.NotNil(t, chat)
			assert.Equal(t, row.stripped || row.unset, chat.Temperature == nil && chat.TopP == nil, "chat wire")
			assert.Equal(t, row.stripped, schemas.AdaptationsFromContext(ctx).Has(schemas.AdaptationSampling), "chat bit")

			ctx = samplingCtx(t)
			resp := ToOpenAIResponsesRequest(ctx, &schemas.BifrostResponsesRequest{
				Provider: row.provider, Model: row.model,
				Input:  []schemas.ResponsesMessage{{Role: schemas.Ptr(schemas.ResponsesInputMessageRoleUser), Content: &schemas.ResponsesMessageContent{ContentStr: user}}},
				Params: &schemas.ResponsesParameters{Temperature: temp, TopP: topP, Reasoning: responsesReasoning(effort)},
			})
			require.NotNil(t, resp)
			assert.Equal(t, row.stripped || row.unset, resp.Temperature == nil && resp.TopP == nil, "responses wire")
			assert.Equal(t, row.stripped, schemas.AdaptationsFromContext(ctx).Has(schemas.AdaptationSampling), "responses bit")

			s := ResponsesSamplingStrip(schemas.ResolveModelCaps(row.provider, row.model), row.provider, row.model, row.effort)
			if !row.unset {
				assert.Equal(t, row.stripped, s.Temperature && s.TopP, "exported decision")
			}
		})
	}
}

func samplingCtx(t *testing.T) *schemas.BifrostContext {
	ctx := schemas.NewBifrostContext(context.Background(), time.Time{})
	t.Cleanup(ctx.Cancel)
	return ctx
}

func chatReasoning(effort *string) *schemas.ChatReasoning {
	if effort == nil {
		return nil
	}
	return &schemas.ChatReasoning{Effort: effort}
}

func responsesReasoning(effort *string) *schemas.ResponsesParametersReasoning {
	if effort == nil {
		return nil
	}
	return &schemas.ResponsesParametersReasoning{Effort: effort}
}

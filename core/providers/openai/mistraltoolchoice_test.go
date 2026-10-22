package openai

import (
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

// Mistral accepts a named function tool_choice in OpenAI's object form, so a
// forced tool stays forced: the converter keeps it rather than widening it to
// "any" (which lets the model call any tool). allowed_tools, which Mistral has
// no form for, becomes the mode's string over the allowed subset of tools.
// The caller's request is never written (BIT-2741 O1).
func TestMistralToolChoice_KeepsForcedChoice(t *testing.T) {
	tools := []schemas.ChatTool{
		{Type: schemas.ChatToolTypeFunction, Function: &schemas.ChatToolFunction{Name: "f"}},
		{Type: schemas.ChatToolTypeFunction, Function: &schemas.ChatToolFunction{Name: "g"}},
	}
	allowed := func(mode string) *schemas.ChatToolChoice {
		return &schemas.ChatToolChoice{ChatToolChoiceStruct: &schemas.ChatToolChoiceStruct{
			Type: schemas.ChatToolChoiceTypeAllowedTools,
			AllowedTools: &schemas.ChatToolChoiceAllowedTools{Mode: mode, Tools: []schemas.ChatToolChoiceAllowedToolsTool{
				{Type: "function", Function: schemas.ChatToolChoiceFunction{Name: "g"}},
			}},
		}}
	}
	for _, tc := range []struct {
		name      string
		provider  schemas.ModelProvider
		model     string
		choice    *schemas.ChatToolChoice
		want      string
		wantTools []string
	}{
		{
			name: "named/mistral", provider: schemas.Mistral, model: "mistral-large-latest",
			choice: &schemas.ChatToolChoice{ChatToolChoiceStruct: &schemas.ChatToolChoiceStruct{
				Type: schemas.ChatToolChoiceTypeFunction, Function: &schemas.ChatToolChoiceFunction{Name: "g"},
			}},
			want: `{"type":"function","function":{"name":"g"}}`, wantTools: []string{"f", "g"},
		},
		{
			name: "named/vertex-mistral", provider: schemas.Vertex, model: "mistral-large-2411",
			choice: &schemas.ChatToolChoice{ChatToolChoiceStruct: &schemas.ChatToolChoiceStruct{
				Type: schemas.ChatToolChoiceTypeFunction, Function: &schemas.ChatToolChoiceFunction{Name: "g"},
			}},
			want: `{"type":"function","function":{"name":"g"}}`, wantTools: []string{"f", "g"},
		},
		{name: "allowed_tools/required", provider: schemas.Mistral, model: "mistral-large-latest", choice: allowed("required"), want: `"any"`, wantTools: []string{"g"}},
		{name: "allowed_tools/auto", provider: schemas.Mistral, model: "mistral-large-latest", choice: allowed("auto"), want: `"auto"`, wantTools: []string{"g"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := chatToolChoiceRequest(tc.provider, tc.model, tc.choice)
			req.Params.Tools = append([]schemas.ChatTool(nil), tools...)
			before := marshalToolChoice(t, tc.choice)

			result := ToOpenAIChatRequest(schemas.NewBifrostContext(nil, schemas.NoDeadline), req)

			if got := marshalToolChoice(t, result.ToolChoice); got != tc.want {
				t.Fatalf("tool_choice on the wire = %s, want %s", got, tc.want)
			}
			var names []string
			for _, tool := range result.Tools {
				names = append(names, tool.Function.Name)
			}
			if len(names) != len(tc.wantTools) {
				t.Fatalf("tools on the wire = %v, want %v", names, tc.wantTools)
			}
			for i := range names {
				if names[i] != tc.wantTools[i] {
					t.Fatalf("tools on the wire = %v, want %v", names, tc.wantTools)
				}
			}
			if after := marshalToolChoice(t, tc.choice); after != before {
				t.Fatalf("caller's tool_choice was written: %s -> %s", before, after)
			}
			if len(req.Params.Tools) != 2 {
				t.Fatalf("caller's tools were written: %d", len(req.Params.Tools))
			}
		})
	}
}

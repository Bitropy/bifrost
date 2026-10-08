package anthropic

import (
	"testing"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/core/schemas"
)

// Anthropic requires input_schema on every custom tool. A chat function tool
// without parameters, or with a raw `{}`, still sends one. A root without
// `type` is carried as sent: no `type` is injected, and a root
// oneOf/anyOf/allOf is rewritten by normalizeAnthropicToolInputSchema exactly
// as for a typed root. Every other schema is converted as before.
func TestConvertFunctionToolToAnthropic_InputSchema(t *testing.T) {
	for _, tc := range []struct {
		name string
		tool string // the OpenAI chat tool as a client sends it
		want string // its input_schema; "" = absent
	}{
		{"parameters_absent", `{"type":"function","function":{"name":"g"}}`, `{"type":"object","properties":{}}`},
		{"parameters_empty_object", `{"type":"function","function":{"name":"g","parameters":{}}}`, `{}`},
		{"parameters_type_only", `{"type":"function","function":{"name":"g","parameters":{"type":"object"}}}`, `{"type":"object","properties":{}}`},
		{"parameters_full", `{"type":"function","function":{"name":"g","parameters":{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}}}`,
			`{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}`},
		// Typeless roots: carried as sent, no `type` injected.
		{"root_description_only", `{"type":"function","function":{"name":"g","parameters":{"description":"d"}}}`, `{"description":"d"}`},
		{"root_bare_properties", `{"type":"function","function":{"name":"g","parameters":{"properties":{"a":{"type":"string"}},"required":["a"]}}}`,
			`{"properties":{"a":{"type":"string"}},"required":["a"]}`},
		// Typeless root compositions: carried, then rewritten into the object
		// schema Anthropic accepts, as a typed root composition is.
		{"root_anyOf", `{"type":"function","function":{"name":"g","parameters":{"anyOf":[{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]},{"type":"object","properties":{"b":{"type":"integer"}},"required":["b"]}]}}}`,
			`{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"integer"}}}`},
		{"root_oneOf", `{"type":"function","function":{"name":"g","parameters":{"oneOf":[{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]},{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"integer"}},"required":["a","b"]}]}}}`,
			`{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"integer"}},"required":["a"]}`},
		{"root_allOf", `{"type":"function","function":{"name":"g","parameters":{"allOf":[{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]},{"type":"object","properties":{"b":{"type":"integer"}}}]}}}`,
			`{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"integer"}},"required":["a"]}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var tool schemas.ChatTool
			if err := sonic.Unmarshal([]byte(tc.tool), &tool); err != nil {
				t.Fatalf("decode tool: %v", err)
			}
			before, err := sonic.Marshal(tool.Function.Parameters)
			if err != nil {
				t.Fatalf("marshal parameters: %v", err)
			}
			converted, err := convertFunctionToolToAnthropic(tool)
			if err != nil {
				t.Fatalf("convert: %v", err)
			}
			// The caller's parameters are not mutated by the conversion.
			if after, err := sonic.Marshal(tool.Function.Parameters); err != nil || string(after) != string(before) {
				t.Errorf("parameters after convert = %s (err %v), want %s", after, err, before)
			}
			wire, err := sonic.Marshal(converted)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			var got struct {
				InputSchema sonic.NoCopyRawMessage `json:"input_schema"`
			}
			if err := sonic.Unmarshal(wire, &got); err != nil {
				t.Fatalf("decode wire: %v", err)
			}
			if string(got.InputSchema) != tc.want {
				t.Errorf("input_schema = %s, want %s (tool on the wire: %s)", got.InputSchema, tc.want, wire)
			}
			// The input_schema on the wire survives a decode/encode round trip
			// unchanged, so a re-serialized request sends the same bytes.
			var params schemas.ToolFunctionParameters
			if err := sonic.Unmarshal(got.InputSchema, &params); err != nil {
				t.Fatalf("decode input_schema: %v", err)
			}
			again, err := sonic.Marshal(params)
			if err != nil {
				t.Fatalf("re-marshal input_schema: %v", err)
			}
			if string(again) != tc.want {
				t.Errorf("input_schema round trip = %s, want %s", again, tc.want)
			}
		})
	}
}

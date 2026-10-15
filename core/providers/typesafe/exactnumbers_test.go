package typesafe

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/bytedance/sonic"
)

// TestDecodeNativeRequestKeepsExactNumbers pins that a native systemone body
// decoded through encoding/json keeps every number in state, instructions and
// criteria as json.Number with its exact digits (BIT-2567: v1.11.3 had no
// custom UnmarshalJSON, so a UseNumber decode kept them; bitrouter's native
// /typesafe/v1/systemone route relies on it), and that the native round trip
// writes those digits back unchanged.
func TestDecodeNativeRequestKeepsExactNumbers(t *testing.T) {
	const body = `{"model":"m","state":{"a":12.50,"id":12345678901234567890},` +
		`"questions":{"q":{"kind":"noul","instructions":{"n":1.0},"criteria":{"true":"x","false":"y"}}}}`

	var req TypesafeDecisionRequest
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	state, ok := req.State.(map[string]interface{})
	if !ok {
		t.Fatalf("state = %#v, want an object", req.State)
	}
	if got := state["a"]; got != json.Number("12.50") {
		t.Errorf(`state.a = %#v, want json.Number("12.50")`, got)
	}
	if got := state["id"]; got != json.Number("12345678901234567890") {
		t.Errorf(`state.id = %#v, want json.Number("12345678901234567890")`, got)
	}
	question, ok := req.Questions["q"]
	if !ok {
		t.Fatalf("question q missing: %#v", req.Questions)
	}
	if question.Type != TypesafeQuestionTypeNoul {
		t.Errorf("question type = %q, want noul (read from kind)", question.Type)
	}
	instructions, ok := question.Instructions.(map[string]interface{})
	if !ok {
		t.Fatalf("instructions = %#v, want an object", question.Instructions)
	}
	if got := instructions["n"]; got != json.Number("1.0") {
		t.Errorf(`instructions.n = %#v, want json.Number("1.0")`, got)
	}
	if !req.stateSet {
		t.Error("state present on decode must be recorded")
	}
	if req.ExtraParams != nil {
		t.Errorf("no extension keys in the body, got ExtraParams %v", req.ExtraParams)
	}

	shared, err := req.ToBifrostDecisionRequest(nil)
	if err != nil {
		t.Fatalf("to shared: %v", err)
	}
	native, err := ToTypesafeDecisionRequest(shared)
	if err != nil {
		t.Fatalf("to native: %v", err)
	}
	wire, err := sonic.Marshal(native)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	for _, digits := range []string{`"a":12.50`, `"id":12345678901234567890`, `"n":1.0`} {
		if !strings.Contains(string(wire), digits) {
			t.Errorf("wire body lost %s: %s", digits, wire)
		}
	}
}

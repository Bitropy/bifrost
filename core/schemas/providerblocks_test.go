package schemas

import (
	"encoding/json"
	"testing"
)

// A prior assistant turn can consist of nothing but provider blocks (a
// pause_turn answer of server-tool blocks); decoding must keep it.
func TestChatMessageUnmarshal_KeepsProviderBlocksOnly(t *testing.T) {
	raw := `{"role":"assistant","provider_blocks":[{"index":1,"provider":"anthropic","block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"q"}}}]}`
	var msg ChatMessage
	if err := Unmarshal([]byte(raw), &msg); err != nil {
		t.Fatal(err)
	}
	if msg.ChatAssistantMessage == nil || len(msg.ChatAssistantMessage.ProviderBlocks) != 1 {
		t.Fatalf("provider_blocks lost on decode: %+v", msg.ChatAssistantMessage)
	}
	pb := msg.ChatAssistantMessage.ProviderBlocks[0]
	if pb.Index != 1 || pb.Provider != ProviderBlockAnthropic {
		t.Errorf("entry = %+v", pb)
	}
	if want := `{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"q"}}`; string(pb.Block) != want {
		t.Errorf("block = %s, want %s", pb.Block, want)
	}
	out, err := Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}
	var back map[string]json.RawMessage
	if err := json.Unmarshal(out, &back); err != nil {
		t.Fatal(err)
	}
	if _, ok := back["provider_blocks"]; !ok {
		t.Errorf("provider_blocks missing on re-encode: %s", out)
	}
}

func TestDeepCopyChatMessage_ProviderBlocksShareNothing(t *testing.T) {
	orig := ChatMessage{
		Role: ChatMessageRoleAssistant,
		ChatAssistantMessage: &ChatAssistantMessage{
			ProviderBlocks: []ChatProviderBlock{{Index: 2, Provider: ProviderBlockAnthropic, Block: json.RawMessage(`{"type":"x"}`)}},
		},
	}
	cp := DeepCopyChatMessage(orig)
	if cp.ChatAssistantMessage == nil || len(cp.ChatAssistantMessage.ProviderBlocks) != 1 {
		t.Fatalf("copy lost provider_blocks: %+v", cp.ChatAssistantMessage)
	}
	cp.ChatAssistantMessage.ProviderBlocks[0].Index = 9
	cp.ChatAssistantMessage.ProviderBlocks[0].Block[2] = 'X'
	if got := orig.ChatAssistantMessage.ProviderBlocks[0]; got.Index != 2 || string(got.Block) != `{"type":"x"}` {
		t.Errorf("copy shares memory with the original: %+v %s", got, got.Block)
	}
}

func TestToolUsage_WebFetchCopiedAndAdded(t *testing.T) {
	u := &ToolUsage{WebFetch: &WebFetchToolUsage{NumRequests: 2}}
	cp := u.DeepCopy()
	cp.WebFetch.NumRequests = 7
	if u.WebFetch.NumRequests != 2 {
		t.Error("DeepCopy shares WebFetch")
	}
	sum := u.Add(&ToolUsage{WebFetch: &WebFetchToolUsage{NumRequests: 3}})
	if sum.WebFetch == nil || sum.WebFetch.NumRequests != 5 {
		t.Errorf("Add web_fetch = %+v", sum.WebFetch)
	}
	out, err := Marshal(u)
	if err != nil {
		t.Fatal(err)
	}
	if string(out) != `{"web_fetch":{"num_requests":2}}` {
		t.Errorf("encoded = %s", out)
	}
}

func TestDeepCopyChatMessage_ProviderCitationShareNothing(t *testing.T) {
	start, end := 1, 2
	orig := ChatMessage{Role: ChatMessageRoleAssistant, ChatAssistantMessage: &ChatAssistantMessage{
		Annotations: []ChatAssistantMessageAnnotation{{
			Type: ChatAnnotationTypeProviderCitation, Provider: "anthropic", StartIndex: &start, EndIndex: &end, Citation: json.RawMessage(`{"a":1}`),
		}},
	}}
	cp := DeepCopyChatMessage(orig).ChatAssistantMessage.Annotations[0]
	if cp.Provider != "anthropic" || cp.StartIndex == nil || *cp.StartIndex != 1 || cp.EndIndex == nil || *cp.EndIndex != 2 || string(cp.Citation) != `{"a":1}` {
		t.Fatalf("copy lost provider citation members: %+v", cp)
	}
	*cp.StartIndex = 9
	cp.Citation[1] = 'X'
	if start != 1 || string(orig.ChatAssistantMessage.Annotations[0].Citation) != `{"a":1}` {
		t.Error("copy shares memory with the original")
	}
}

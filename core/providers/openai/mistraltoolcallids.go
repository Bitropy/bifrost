package openai

import (
	"crypto/sha256"
	"strconv"

	"github.com/maximhq/bifrost/core/schemas"
)

// mistralToolCallIDLength is the only tool call id length Mistral accepts.
const mistralToolCallIDLength = 9

// mistralToolCallIDAlphabet is the alphabet Mistral accepts in a tool call id.
const mistralToolCallIDAlphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

// validMistralToolCallID reports whether id matches ^[a-zA-Z0-9]{9}$.
func validMistralToolCallID(id string) bool {
	if len(id) != mistralToolCallIDLength {
		return false
	}
	for i := 0; i < len(id); i++ {
		c := id[i]
		if (c < '0' || c > '9') && (c < 'a' || c > 'z') && (c < 'A' || c > 'Z') {
			return false
		}
	}
	return true
}

// mistralToolCallID derives a nine-alphanumeric id from id and a collision
// counter: the first nine bytes of SHA-256(id, NUL, counter), each mapped onto
// the alphabet.
func mistralToolCallID(id string, counter int) string {
	h := sha256.New()
	h.Write([]byte(id))
	h.Write([]byte{0})
	h.Write(strconv.AppendInt(nil, int64(counter), 10))
	sum := h.Sum(nil)
	out := make([]byte, mistralToolCallIDLength)
	for i := range out {
		out[i] = mistralToolCallIDAlphabet[int(sum[i])%len(mistralToolCallIDAlphabet)]
	}
	return string(out)
}

// rewriteMistralToolCallIDs rewrites every tool call id Mistral would refuse
// ("Tool call id was call_x but must be a-z, A-Z, 0-9, with a length of 9",
// a 400) into nine alphanumerics, so a history replayed from another model
// (OpenAI call_…, Anthropic toolu_…, an earlier attempt of a failover chain)
// is served.
//
//   - Deterministic per request: an id maps to a hash of itself, so a retry,
//     a failover attempt and the next turn of the conversation send the same
//     ids (the prompt cache sees one prefix).
//   - Consistent: an assistant tool_calls[].id and the tool message's
//     tool_call_id answering it map through the same table.
//   - Collision-free: valid ids are kept as they are and reserved first; a
//     derived id that collides with any id already in use is derived again
//     with the next counter.
//
// Request-side only. The ids rewritten here exist only in the history the
// client already holds under its own ids; Mistral's answer carries only the
// ids of its new tool calls, which Mistral generates in its own (valid)
// shape, so the client never sees a rewritten id and nothing maps back.
//
// The caller's request is never written: a rewritten tool message gets its
// own ChatToolMessage, and a rewritten assistant message its own tool_calls
// slice (both are shared with the caller's input).
func (req *OpenAIChatRequest) rewriteMistralToolCallIDs() {
	used := map[string]struct{}{}
	var invalid []string
	note := func(id *string) {
		if id == nil {
			return
		}
		if validMistralToolCallID(*id) {
			used[*id] = struct{}{}
			return
		}
		invalid = append(invalid, *id)
	}
	for i := range req.Messages {
		m := &req.Messages[i]
		if m.ChatToolMessage != nil {
			note(m.ChatToolMessage.ToolCallID)
		}
		if m.OpenAIChatAssistantMessage != nil {
			for j := range m.OpenAIChatAssistantMessage.ToolCalls {
				note(m.OpenAIChatAssistantMessage.ToolCalls[j].ID)
			}
		}
	}
	if len(invalid) == 0 {
		return
	}
	mapped := make(map[string]string, len(invalid))
	for _, id := range invalid {
		if _, ok := mapped[id]; ok {
			continue
		}
		for counter := 0; ; counter++ {
			candidate := mistralToolCallID(id, counter)
			if _, taken := used[candidate]; !taken {
				used[candidate] = struct{}{}
				mapped[id] = candidate
				break
			}
		}
	}
	for i := range req.Messages {
		m := &req.Messages[i]
		if tm := m.ChatToolMessage; tm != nil && tm.ToolCallID != nil {
			if to, ok := mapped[*tm.ToolCallID]; ok {
				cp := *tm
				cp.ToolCallID = &to
				m.ChatToolMessage = &cp
			}
		}
		if am := m.OpenAIChatAssistantMessage; am != nil {
			m.OpenAIChatAssistantMessage.ToolCalls = rewriteToolCallIDs(am.ToolCalls, mapped)
		}
	}
}

// rewriteToolCallIDs returns calls with every id in mapped replaced, in a new
// slice when anything changes and calls itself otherwise.
func rewriteToolCallIDs(calls []schemas.ChatAssistantMessageToolCall, mapped map[string]string) []schemas.ChatAssistantMessageToolCall {
	var out []schemas.ChatAssistantMessageToolCall
	for j := range calls {
		if calls[j].ID == nil {
			continue
		}
		to, ok := mapped[*calls[j].ID]
		if !ok {
			continue
		}
		if out == nil {
			out = make([]schemas.ChatAssistantMessageToolCall, len(calls))
			copy(out, calls)
		}
		out[j].ID = &to
	}
	if out == nil {
		return calls
	}
	return out
}

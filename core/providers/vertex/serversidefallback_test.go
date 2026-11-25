package vertex

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

// vertexFallbackMessage is a unary Claude answer whose usage names a server-side fallback model.
const vertexFallbackMessage = `{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5",` +
	`"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn",` +
	`"usage":{"input_tokens":7,"output_tokens":5,"iterations":[{"type":"fallback_message","model":"claude-opus-4-8","input_tokens":7,"output_tokens":5}]}}`

// AC-4: the Vertex unary Claude branches of chat and Responses keep the served dimension
// RoutingInfo.ServerSideFallbackModel that the Anthropic converters set.
func TestVertexUnary_KeepsServerSideFallbackModel(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, vertexFallbackMessage)
	}))
	t.Cleanup(srv.Close)
	provider, key := newClaudeTestVertex(t, srv.URL)
	const model = "claude-sonnet-4-5"
	user := schemas.ChatMessage{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("hi")}}

	check := func(t *testing.T, got *string) {
		t.Helper()
		if got == nil || *got != "claude-opus-4-8" {
			t.Fatalf("ServerSideFallbackModel = %v, want claude-opus-4-8", got)
		}
	}
	t.Run("chat", func(t *testing.T) {
		ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
		resp, bifrostErr := provider.ChatCompletion(ctx, key, &schemas.BifrostChatRequest{
			Provider: schemas.Vertex, Model: model, Input: []schemas.ChatMessage{user},
			Params: &schemas.ChatParameters{MaxCompletionTokens: schemas.Ptr(64)},
		})
		if bifrostErr != nil {
			t.Fatal(bifrostErr.GetErrorString())
		}
		check(t, resp.ExtraFields.RoutingInfo.ServerSideFallbackModel)
	})
	t.Run("responses", func(t *testing.T) {
		ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
		resp, bifrostErr := provider.Responses(ctx, key, &schemas.BifrostResponsesRequest{
			Provider: schemas.Vertex, Model: model,
			Input:  []schemas.ResponsesMessage{{Role: schemas.Ptr(schemas.ResponsesInputMessageRoleUser), Content: &schemas.ResponsesMessageContent{ContentStr: schemas.Ptr("hi")}}},
			Params: &schemas.ResponsesParameters{MaxOutputTokens: schemas.Ptr(64)},
		})
		if bifrostErr != nil {
			t.Fatal(bifrostErr.GetErrorString())
		}
		check(t, resp.ExtraFields.RoutingInfo.ServerSideFallbackModel)
	})
}

// Package wirecapture is a test helper for the request bodies core writes to an upstream. Used by
// tests only.
//
// Generic: Server / Next (a capture server answering 400), Context, Drain, PostHookRunner, JSON and
// Logger. BIT-2906-specific: the PNG constants, the ChatToolResult* / ResponsesToolOutputImage
// fixtures and the Assert*PlaceholderSizeFreeUnderRewriter checks, which require that a
// request-body rewriter on the context changes a tool-result media placeholder only by leaving its
// size out.
package wirecapture

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
)

// The tool-result image every case sends, and the placeholder a text-only wire gets for it.
const (
	PNGDataURL  = "data:image/png;base64,AAAAAAAAAAAA"
	SizedPNG    = "[image/png, 9 bytes returned by the tool, not shown]"
	SizeFreePNG = "[image/png returned by the tool, not shown]"
)

// Server answers every request with a 400 and hands each request body to Next.
type Server struct {
	URL    string
	bodies chan []byte
}

// NewServer starts a capture server, closed when t ends.
func NewServer(t testing.TB) *Server {
	t.Helper()
	s := &Server{bodies: make(chan []byte, 16)}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		s.bodies <- body
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"message":"captured","error":{"message":"captured","type":"invalid_request_error"}}`)
	}))
	t.Cleanup(srv.Close)
	s.URL = srv.URL
	return s
}

// Next is the next captured request body.
func (s *Server) Next(t testing.TB) []byte {
	t.Helper()
	select {
	case body := <-s.bodies:
		return body
	case <-time.After(5 * time.Second):
		t.Fatal("no request reached the capture server")
		return nil
	}
}

// Context is a request context, carrying a pass-through request-body rewriter when rewriter is set.
func Context(rewriter bool) *schemas.BifrostContext {
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	if rewriter {
		ctx.SetValue(schemas.BifrostContextKeyRequestBodyRewriter, providerUtils.RequestBodyRewriter(func([]byte) (io.Reader, int64, error) { return nil, 0, nil }))
	}
	return ctx
}

// Drain reads a stream to its end; a nil channel (a request refused before streaming) is a no-op.
func Drain(chunks chan *schemas.BifrostStreamChunk) {
	for chunks != nil {
		if _, open := <-chunks; !open {
			return
		}
	}
}

// PostHookRunner passes results through unchanged.
func PostHookRunner(_ *schemas.BifrostContext, r *schemas.BifrostResponse, e *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError) {
	return r, e
}

// AssertPlaceholderSizeFreeUnderRewriter sends once without and once with a rewriter (send returns
// the wire body) and requires upstream's sized placeholder without one, and a body that differs only
// by the size with one.
func AssertPlaceholderSizeFreeUnderRewriter(t *testing.T, sized, sizeFree string, send func(ctx *schemas.BifrostContext) []byte) {
	t.Helper()
	plain := send(Context(false))
	if !bytes.Contains(plain, []byte(sized)) {
		t.Fatalf("without a rewriter the placeholder must be upstream's %q; body:\n%s", sized, plain)
	}
	want := bytes.ReplaceAll(plain, []byte(sized), []byte(sizeFree))
	if got := send(Context(true)); !bytes.Equal(got, want) {
		t.Fatalf("under a rewriter only the size may leave the placeholder\nwant %s\ngot  %s", want, got)
	}
}

// ChatToolResult is a chat conversation whose tool result is block alone.
func ChatToolResult(block schemas.ChatContentBlock) []schemas.ChatMessage {
	return []schemas.ChatMessage{
		{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("look")}},
		{Role: schemas.ChatMessageRoleAssistant, ChatAssistantMessage: &schemas.ChatAssistantMessage{ToolCalls: []schemas.ChatAssistantMessageToolCall{
			{ID: schemas.Ptr("call_1"), Type: schemas.Ptr("function"), Function: schemas.ChatAssistantMessageToolCallFunction{Name: schemas.Ptr("shot"), Arguments: "{}"}},
		}}},
		{Role: schemas.ChatMessageRoleTool, ChatToolMessage: &schemas.ChatToolMessage{ToolCallID: schemas.Ptr("call_1")}, Content: &schemas.ChatMessageContent{ContentBlocks: []schemas.ChatContentBlock{block}}},
	}
}

// ChatToolResultImage is ChatToolResult with an image of the data URL url.
func ChatToolResultImage(url string) []schemas.ChatMessage {
	return ChatToolResult(schemas.ChatContentBlock{Type: schemas.ChatContentBlockTypeImage, ImageURLStruct: &schemas.ChatInputImage{URL: url}})
}

// ResponsesToolOutputImage is a Responses input whose function_call_output is an image of the data URL url.
func ResponsesToolOutputImage(url string) []schemas.ResponsesMessage {
	return []schemas.ResponsesMessage{
		{Type: schemas.Ptr(schemas.ResponsesMessageTypeMessage), Role: schemas.Ptr(schemas.ResponsesInputMessageRoleUser), Content: &schemas.ResponsesMessageContent{ContentStr: schemas.Ptr("look")}},
		{Type: schemas.Ptr(schemas.ResponsesMessageTypeFunctionCall), ResponsesToolMessage: &schemas.ResponsesToolMessage{CallID: schemas.Ptr("c1"), Name: schemas.Ptr("shot"), Arguments: schemas.Ptr(`{}`)}},
		{Type: schemas.Ptr(schemas.ResponsesMessageTypeFunctionCallOutput), ResponsesToolMessage: &schemas.ResponsesToolMessage{
			CallID: schemas.Ptr("c1"),
			Output: &schemas.ResponsesToolMessageOutputStruct{ResponsesFunctionToolCallOutputBlocks: []schemas.ResponsesMessageContentBlock{
				{Type: schemas.ResponsesInputMessageContentBlockTypeImage, ResponsesInputMessageContentBlockImage: &schemas.ResponsesInputMessageContentBlockImage{ImageURL: schemas.Ptr(url)}},
			}},
		}},
	}
}

// JSON is v marshalled as a provider sends it.
func JSON(t testing.TB, v any) []byte {
	t.Helper()
	body, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return body
}

// Logger discards every log line; for provider constructors in packages that cannot import core.
type Logger struct{}

func (Logger) Debug(string, ...any)                                            {}
func (Logger) Info(string, ...any)                                             {}
func (Logger) Warn(string, ...any)                                             {}
func (Logger) Error(string, ...any)                                            {}
func (Logger) Fatal(string, ...any)                                            {}
func (Logger) SetLevel(schemas.LogLevel)                                       {}
func (Logger) SetOutputType(schemas.LoggerOutputType)                          {}
func (Logger) LogHTTPRequest(schemas.LogLevel, string) schemas.LogEventBuilder { return eventBuilder{} }

type eventBuilder struct{}

func (b eventBuilder) Str(string, string) schemas.LogEventBuilder  { return b }
func (b eventBuilder) Int(string, int) schemas.LogEventBuilder     { return b }
func (b eventBuilder) Int64(string, int64) schemas.LogEventBuilder { return b }
func (eventBuilder) Send()                                         {}

// ChatProvider is the chat half of schemas.Provider.
type ChatProvider interface {
	ChatCompletion(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostChatRequest) (*schemas.BifrostChatResponse, *schemas.BifrostError)
	ChatCompletionStream(ctx *schemas.BifrostContext, postHookRunner schemas.PostHookRunner, postHookSpanFinalizer func(context.Context), key schemas.Key, request *schemas.BifrostChatRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError)
}

// ResponsesProvider is the Responses half of schemas.Provider.
type ResponsesProvider interface {
	Responses(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostResponsesRequest) (*schemas.BifrostResponsesResponse, *schemas.BifrostError)
	ResponsesStream(ctx *schemas.BifrostContext, postHookRunner schemas.PostHookRunner, postHookSpanFinalizer func(context.Context), key schemas.Key, request *schemas.BifrostResponsesRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError)
}

// AssertChatPlaceholderSizeFreeUnderRewriter runs AssertPlaceholderSizeFreeUnderRewriter on the
// PNG placeholder for a unary and a streaming chat request, each captured by s.
func AssertChatPlaceholderSizeFreeUnderRewriter(t *testing.T, s *Server, p ChatProvider, key schemas.Key, request func() *schemas.BifrostChatRequest) {
	t.Helper()
	assertUnaryAndStream(t, s,
		func(ctx *schemas.BifrostContext) { _, _ = p.ChatCompletion(ctx, key, request()) },
		func(ctx *schemas.BifrostContext) {
			chunks, _ := p.ChatCompletionStream(ctx, PostHookRunner, nil, key, request())
			Drain(chunks)
		})
}

// AssertResponsesPlaceholderSizeFreeUnderRewriter is AssertChatPlaceholderSizeFreeUnderRewriter for
// Responses requests.
func AssertResponsesPlaceholderSizeFreeUnderRewriter(t *testing.T, s *Server, p ResponsesProvider, key schemas.Key, request func() *schemas.BifrostResponsesRequest) {
	t.Helper()
	assertUnaryAndStream(t, s,
		func(ctx *schemas.BifrostContext) { _, _ = p.Responses(ctx, key, request()) },
		func(ctx *schemas.BifrostContext) {
			chunks, _ := p.ResponsesStream(ctx, PostHookRunner, nil, key, request())
			Drain(chunks)
		})
}

func assertUnaryAndStream(t *testing.T, s *Server, unary, stream func(ctx *schemas.BifrostContext)) {
	t.Helper()
	for name, send := range map[string]func(ctx *schemas.BifrostContext){"unary": unary, "stream": stream} {
		t.Run(name, func(t *testing.T) {
			AssertPlaceholderSizeFreeUnderRewriter(t, SizedPNG, SizeFreePNG, func(ctx *schemas.BifrostContext) []byte {
				send(ctx)
				return s.Next(t)
			})
		})
	}
}

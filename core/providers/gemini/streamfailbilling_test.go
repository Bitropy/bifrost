package gemini

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
)

// A failed Gemini chat stream bills what the provider already reported: the
// running usage handle follows every usage-bearing event (usageMetadata is
// cumulative), not only the final chunk, and the error exits attach it with
// the served tier as BilledUsage, like the cancel and timeout exits (BIT-2741
// F8-2).

const geminiTieredContent = `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hel"}]},"index":0}],` +
	`"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":3,"totalTokenCount":15,"trafficType":"ON_DEMAND_PRIORITY"},` +
	`"modelVersion":"gemini-2.5-flash"}` + "\n\n"

const geminiErrorEvent = `data: {"error":{"code":503,"message":"The model is overloaded.","status":"UNAVAILABLE"}}` + "\n\n"

type geminiTestLogger struct{}

func (geminiTestLogger) Debug(string, ...any)                   {}
func (geminiTestLogger) Info(string, ...any)                    {}
func (geminiTestLogger) Warn(string, ...any)                    {}
func (geminiTestLogger) Error(string, ...any)                   {}
func (geminiTestLogger) Fatal(string, ...any)                   {}
func (geminiTestLogger) SetLevel(schemas.LogLevel)              {}
func (geminiTestLogger) SetOutputType(schemas.LoggerOutputType) {}
func (geminiTestLogger) LogHTTPRequest(schemas.LogLevel, string) schemas.LogEventBuilder {
	return schemas.NoopLogEvent
}

type geminiErrorCapture struct {
	mu   sync.Mutex
	errs []*schemas.BifrostError
}

func (c *geminiErrorCapture) hook(_ *schemas.BifrostContext, resp *schemas.BifrostResponse, err *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError) {
	if err != nil {
		c.mu.Lock()
		c.errs = append(c.errs, err)
		c.mu.Unlock()
	}
	return resp, err
}

func (c *geminiErrorCapture) only(t *testing.T) *schemas.BifrostError {
	t.Helper()
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.errs) != 1 {
		t.Fatalf("expected exactly one stream error, got %d: %+v", len(c.errs), c.errs)
	}
	return c.errs[0]
}

// geminiStreamServer writes prelude; then either ends (malformed=false) or
// writes a chunk header that is not hex so the body read fails (malformed=true).
func geminiStreamServer(t *testing.T, prelude string, malformed bool) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !malformed {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(prelude))
			return
		}
		conn, buf, err := w.(http.Hijacker).Hijack()
		if err != nil {
			t.Errorf("hijack: %v", err)
			return
		}
		defer conn.Close()
		_, _ = buf.WriteString("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
		_, _ = fmt.Fprintf(buf, "%x\r\n%s\r\n", len(prelude), prelude)
		_, _ = buf.WriteString("zz\r\n")
		_ = buf.Flush()
	}))
	t.Cleanup(server.Close)
	return server
}

func drainGeminiStream(t *testing.T, stream chan *schemas.BifrostStreamChunk) {
	t.Helper()
	timeout := time.NewTimer(20 * time.Second)
	defer timeout.Stop()
	for {
		select {
		case _, ok := <-stream:
			if !ok {
				return
			}
		case <-timeout.C:
			t.Fatal("timed out waiting for the provider stream to close")
		}
	}
}

func TestGeminiChatStreamFailedExitsBillServedUsage(t *testing.T) {
	for _, tc := range []struct {
		name      string
		prelude   string
		malformed bool
	}{
		{name: "error_event", prelude: geminiTieredContent + geminiErrorEvent},
		{name: "read_error", prelude: geminiTieredContent, malformed: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := geminiStreamServer(t, tc.prelude, tc.malformed)
			provider := NewGeminiProvider(&schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{BaseURL: server.URL}}, geminiTestLogger{})
			capture := &geminiErrorCapture{}
			ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
			stream, err := provider.ChatCompletionStream(ctx, capture.hook, nil, schemas.Key{Value: *schemas.NewSecretVar("k")}, &schemas.BifrostChatRequest{
				Provider: schemas.Gemini,
				Model:    "gemini-2.5-flash",
				Input:    []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("hi")}}},
			})
			if err != nil {
				t.Fatalf("stream setup failed: %+v", err)
			}
			drainGeminiStream(t, stream)
			got := capture.only(t)
			billed := got.ExtraFields.BilledUsage
			if billed == nil {
				t.Fatalf("expected the reported usage as BilledUsage on %+v", got.Error)
			}
			if billed.PromptTokens != 12 || billed.CompletionTokens != 3 || billed.TotalTokens != 15 {
				t.Errorf("billed = %d/%d/%d, want 12/3/15", billed.PromptTokens, billed.CompletionTokens, billed.TotalTokens)
			}
			if billed.ServiceTier == nil || *billed.ServiceTier != schemas.BifrostServiceTierPriority {
				t.Errorf("billed service tier = %v, want priority", billed.ServiceTier)
			}
		})
	}
}

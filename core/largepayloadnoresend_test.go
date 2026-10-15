package bifrost

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
)

// largePayloadNoResendBody is the request body the large-payload reader carries.
const largePayloadNoResendBody = `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`

// keyedUpstream records every hit by the key it carried and answers per key: key A gets
// aStatus with aBody, key B gets a chat completion.
type keyedUpstream struct {
	mu   sync.Mutex
	hits []string
}

func (u *keyedUpstream) record(key string) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.hits = append(u.hits, key)
}

func (u *keyedUpstream) snapshot() []string {
	u.mu.Lock()
	defer u.mu.Unlock()
	return append([]string(nil), u.hits...)
}

func newKeyedUpstream(t *testing.T, aStatus int, aBody string) (*keyedUpstream, *httptest.Server) {
	t.Helper()
	u := &keyedUpstream{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		switch r.Header.Get("Authorization") {
		case "Bearer sk-a":
			u.record("key-a")
			w.WriteHeader(aStatus)
			_, _ = io.WriteString(w, aBody)
		case "Bearer sk-b":
			u.record("key-b")
			_, _ = io.WriteString(w, `{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-4o",`+
				`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],`+
				`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
		default:
			u.record("unknown")
			w.WriteHeader(http.StatusBadRequest)
		}
	}))
	t.Cleanup(srv.Close)
	return u, srv
}

// newTwoKeyClient builds a real Bifrost over upstream with keys A and B, a selector that
// always picks the first key it is offered (A while A is live), and maxRetries.
func newTwoKeyClient(t *testing.T, upstreamURL string, maxRetries int) *Bifrost {
	t.Helper()
	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 4, upstreamURL)
	account.configs[schemas.OpenAI].NetworkConfig.MaxRetries = maxRetries
	account.configs[schemas.OpenAI].NetworkConfig.RetryBackoffInitial = time.Millisecond
	account.configs[schemas.OpenAI].NetworkConfig.RetryBackoffMax = 2 * time.Millisecond
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{
		{ID: "key-a", Name: "Key A", Value: *schemas.NewSecretVar("sk-a"), Models: schemas.WhiteList{"*"}, Weight: 1},
		{ID: "key-b", Name: "Key B", Value: *schemas.NewSecretVar("sk-b"), Models: schemas.WhiteList{"*"}, Weight: 1},
	})
	client, err := Init(context.Background(), schemas.BifrostConfig{
		Account: account,
		Logger:  NewDefaultLogger(schemas.LogLevelError),
		KeySelector: func(_ *schemas.BifrostContext, keys []schemas.Key, _ schemas.ModelProvider, _ string) (schemas.Key, error) {
			return keys[0], nil
		},
	})
	if err != nil {
		t.Fatalf("Init: %v", err)
	}
	t.Cleanup(client.Shutdown)
	return client
}

func largePayloadChatRequest() *schemas.BifrostChatRequest {
	return &schemas.BifrostChatRequest{
		Provider: schemas.OpenAI,
		Model:    "gpt-4o",
		Input:    []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("hi")}}},
	}
}

func largePayloadContext() *schemas.BifrostContext {
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	ctx.SetValue(schemas.BifrostContextKeyLargePayloadMode, true)
	ctx.SetValue(schemas.BifrostContextKeyLargePayloadReader, io.Reader(bytes.NewReader([]byte(largePayloadNoResendBody))))
	ctx.SetValue(schemas.BifrostContextKeyLargePayloadContentLength, len(largePayloadNoResendBody))
	ctx.SetValue(schemas.BifrostContextKeyLargePayloadContentType, "application/json")
	return ctx
}

// A large-payload request carries its body in a one-shot reader, so core must never send it
// twice: a second attempt would read an exhausted reader and fail locally, and that local
// failure would replace the provider's real answer. The first attempt's error is the answer,
// whatever its class and whatever max_retries says.
func TestLargePayloadRequestIsNeverResent(t *testing.T) {
	const invalidKey = `{"error":{"message":"Incorrect API key provided","type":"invalid_request_error","code":"invalid_api_key"}}`

	t.Run("permanent per-key failure at max_retries 0 keeps the 401", func(t *testing.T) {
		upstream, srv := newKeyedUpstream(t, http.StatusUnauthorized, invalidKey)
		client := newTwoKeyClient(t, srv.URL, 0)

		ctx := largePayloadContext()
		_, bifrostErr := client.ChatCompletionRequest(ctx, largePayloadChatRequest())
		// A re-send dies locally on the drained reader before reaching the upstream, so the
		// attempt trail, not the hit count alone, is what proves there was no second attempt.
		if trail, _ := ctx.Value(schemas.BifrostContextKeyAttemptTrail).([]schemas.KeyAttemptRecord); len(trail) != 1 {
			t.Fatalf("attempt trail has %d record(s), want 1: %+v", len(trail), trail)
		}
		if hits := upstream.snapshot(); len(hits) != 1 || hits[0] != "key-a" {
			t.Fatalf("upstream hits = %v, want exactly one on key-a", hits)
		}
		if bifrostErr == nil || bifrostErr.StatusCode == nil || *bifrostErr.StatusCode != http.StatusUnauthorized {
			t.Fatalf("want key A's 401, got %+v", bifrostErr)
		}
		if bifrostErr.Error == nil || !strings.Contains(bifrostErr.Error.Message, "Incorrect API key provided") {
			t.Fatalf("want key A's message, got %+v", bifrostErr.Error)
		}
	})

	t.Run("transient failure at max_retries 2 keeps the 500", func(t *testing.T) {
		upstream, srv := newKeyedUpstream(t, http.StatusInternalServerError,
			`{"error":{"message":"upstream exploded","type":"server_error"}}`)
		client := newTwoKeyClient(t, srv.URL, 2)

		ctx := largePayloadContext()
		_, bifrostErr := client.ChatCompletionRequest(ctx, largePayloadChatRequest())
		// A re-send dies locally on the drained reader before reaching the upstream, so the
		// attempt trail, not the hit count alone, is what proves there was no second attempt.
		if trail, _ := ctx.Value(schemas.BifrostContextKeyAttemptTrail).([]schemas.KeyAttemptRecord); len(trail) != 1 {
			t.Fatalf("attempt trail has %d record(s), want 1: %+v", len(trail), trail)
		}
		if hits := upstream.snapshot(); len(hits) != 1 || hits[0] != "key-a" {
			t.Fatalf("upstream hits = %v, want exactly one on key-a", hits)
		}
		if bifrostErr == nil || bifrostErr.StatusCode == nil || *bifrostErr.StatusCode != http.StatusInternalServerError {
			t.Fatalf("want key A's 500, got %+v", bifrostErr)
		}
		if bifrostErr.Error == nil || !strings.Contains(bifrostErr.Error.Message, "upstream exploded") {
			t.Fatalf("want key A's message, got %+v", bifrostErr.Error)
		}
	})

	t.Run("a normal request still rotates past the rejected key", func(t *testing.T) {
		upstream, srv := newKeyedUpstream(t, http.StatusUnauthorized, invalidKey)
		client := newTwoKeyClient(t, srv.URL, 0)

		resp, bifrostErr := client.ChatCompletionRequest(schemas.NewBifrostContext(context.Background(), schemas.NoDeadline), largePayloadChatRequest())
		if bifrostErr != nil {
			t.Fatalf("want key B to serve, got %+v", bifrostErr)
		}
		if resp == nil || len(resp.Choices) == 0 {
			t.Fatalf("want a completion, got %+v", resp)
		}
		if hits := upstream.snapshot(); len(hits) != 2 || hits[0] != "key-a" || hits[1] != "key-b" {
			t.Fatalf("upstream hits = %v, want key-a then key-b", hits)
		}
	})
}

package bifrost

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
)

// billedErrorRecorder keeps, per core attempt, the billed usage of each error its post-hook sees.
type billedErrorRecorder struct {
	endedAttemptRecorder
	mu     sync.Mutex
	billed map[int][]*schemas.BifrostLLMUsage
}

func (r *billedErrorRecorder) PostLLMHook(ctx *schemas.BifrostContext, resp *schemas.BifrostResponse, bifrostErr *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError, error) {
	if bifrostErr != nil {
		attempt, _ := ctx.Value(schemas.BifrostContextKeyNumberOfRetries).(int)
		r.mu.Lock()
		r.billed[attempt] = append(r.billed[attempt], bifrostErr.ExtraFields.BilledUsage.DeepCopy())
		r.mu.Unlock()
	}
	return resp, bifrostErr, nil
}

func (r *billedErrorRecorder) attempt(n int) []*schemas.BifrostLLMUsage {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]*schemas.BifrostLLMUsage(nil), r.billed[n]...)
}

// staleUsageFirstAttempt registers the usage handle, reports usage, then fails with a retryable
// error event before any content reaches the caller: the error bills 12/3.
const staleUsageFirstAttempt = `data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3,"total_tokens":15}}` + "\n\n" +
	`data: {"error":{"message":"The server had an error.","type":"server_error"}}` + "\n\n"

// BIT-2978: core reuses one context for every attempt, so a retry that fails before it registers
// its own usage handle, through an exit that bills from the context, would re-report the previous
// attempt's usage. Attempt 0 fails with billed usage; attempt 1 answers 200 with a non-SSE body,
// either at once (the non-SSE probe) or never, so the request deadline ends it while the probe
// still waits for the first byte (the timeout exit, reached before the handle is registered).
// Attempt 1's errors must carry no billed usage.
func TestExecuteRequestWithRetries_RetryDoesNotReuseStaleUsageHandle(t *testing.T) {
	const provider schemas.ModelProvider = "stale-usage"
	for _, tc := range []struct {
		name  string
		stall bool
	}{
		{name: "non_sse_probe"},
		{name: "timeout_before_handle", stall: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var hits atomic.Int32
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				if hits.Add(1) == 1 {
					w.Header().Set("Content-Type", "text/event-stream")
					fmt.Fprint(w, staleUsageFirstAttempt)
					return
				}
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				if !tc.stall {
					fmt.Fprint(w, `{"error":{"message":"not a stream"}}`)
					return
				}
				w.(http.Flusher).Flush()
				select {
				case <-r.Context().Done():
				case <-time.After(5 * time.Second):
				}
			}))
			t.Cleanup(upstream.Close)

			account := NewMockAccount()
			account.AddProviderWithBaseURL(provider, 1, 16, upstream.URL)
			account.configs[provider].NetworkConfig.MaxRetries = 1
			account.configs[provider].NetworkConfig.RetryBackoffInitial = time.Millisecond
			account.configs[provider].NetworkConfig.RetryBackoffMax = time.Millisecond
			account.SetCustomProviderConfig(provider, &schemas.CustomProviderConfig{BaseProviderType: schemas.OpenAI})
			account.SetKeysForProvider(provider, []schemas.Key{{ID: "k1", Value: *schemas.NewSecretVar("sk-k1"), Models: schemas.WhiteList{"*"}, Weight: 100}})
			recorder := &billedErrorRecorder{billed: map[int][]*schemas.BifrostLLMUsage{}}
			client, err := Init(context.Background(), schemas.BifrostConfig{Account: account, Logger: NewNoOpLogger(), LLMPlugins: []schemas.LLMPlugin{recorder}})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(client.Shutdown)

			deadline := schemas.NoDeadline
			if tc.stall {
				// Attempt 0 and attempt 1's headers take milliseconds; the deadline lands in the probe.
				deadline = time.Now().Add(time.Second)
			}
			ctx := schemas.NewBifrostContext(context.Background(), deadline)
			stream, bifrostErr := client.ChatCompletionStreamRequest(ctx, &schemas.BifrostChatRequest{
				Provider: provider, Model: "m",
				Input: []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: Ptr("hi")}}},
			})
			if bifrostErr == nil {
				for range stream {
				}
			}
			if got := hits.Load(); got != 2 {
				t.Fatalf("upstream hits = %d, want 2 (one retry)", got)
			}

			first := recorder.attempt(0)
			if len(first) == 0 || first[0] == nil || first[0].PromptTokens != 12 || first[0].CompletionTokens != 3 {
				t.Fatalf("attempt 0 errors billed %+v, want 12/3", first)
			}
			var retry []*schemas.BifrostLLMUsage
			for end := time.Now().Add(3 * time.Second); time.Now().Before(end); time.Sleep(10 * time.Millisecond) {
				if retry = recorder.attempt(1); len(retry) > 0 {
					break
				}
			}
			if len(retry) == 0 {
				t.Fatal("no post-hook saw attempt 1's error")
			}
			for _, billed := range retry {
				if billed != nil {
					t.Fatalf("attempt 1 re-billed the previous attempt's usage: %+v", billed)
				}
			}
		})
	}
}

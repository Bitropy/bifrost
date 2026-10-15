package bifrost

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
)

// openAIStyleModelsHandler serves a minimal OpenAI-format GET /v1/models
// response listing the given model IDs.
func openAIStyleModelsHandler(ids ...string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		body := `{"object":"list","data":[`
		for i, id := range ids {
			if i > 0 {
				body += ","
			}
			body += `{"id":"` + id + `","object":"model","created":1700000000,"owned_by":"test"}`
		}
		body += `]}`
		_, _ = w.Write([]byte(body))
	}
}

// wildcardKey returns a key whose allowlist admits every model, so the
// filtered list-models path doesn't early-exit to an empty list.
func wildcardKey(provider schemas.ModelProvider) []schemas.Key {
	return []schemas.Key{{
		ID:     "test-key-" + string(provider),
		Value:  *schemas.NewSecretVar("sk-test-" + string(provider)),
		Models: schemas.WhiteList{"*"},
		Weight: 100,
	}}
}

// tuneProviderForTest shrinks the provider's HTTP budget so an abandoned
// in-flight call to a hanging test server releases its worker (and Shutdown)
// quickly, and disables retries for deterministic single-attempt behavior.
func tuneProviderForTest(t *testing.T, account *MockAccount, provider schemas.ModelProvider) {
	t.Helper()
	cfg, ok := account.configs[provider]
	if !ok {
		t.Fatalf("provider %s not configured on mock account", provider)
	}
	cfg.NetworkConfig.DefaultRequestTimeoutInSeconds = 2
	cfg.NetworkConfig.MaxRetries = 0
}

func newListModelsTestClient(t *testing.T, account *MockAccount, catalog schemas.ModelInfoProvider) *Bifrost {
	t.Helper()
	client, initErr := Init(context.Background(), schemas.BifrostConfig{
		Account:      account,
		Logger:       NewNoOpLogger(),
		ModelCatalog: catalog,
	})
	if initErr != nil {
		t.Fatalf("Init failed: %v", initErr)
	}
	t.Cleanup(client.Shutdown)
	return client
}

func modelIDs(resp *schemas.BifrostListModelsResponse) []string {
	if resp == nil {
		return nil
	}
	ids := make([]string, 0, len(resp.Data))
	for _, m := range resp.Data {
		ids = append(ids, m.ID)
	}
	return ids
}

func containsModelID(resp *schemas.BifrostListModelsResponse, id string) bool {
	if resp == nil {
		return false
	}
	for _, m := range resp.Data {
		if m.ID == id {
			return true
		}
	}
	return false
}

// A single unreachable provider must not block the aggregate list: the healthy
// provider's models should come back well under the unreachable provider's
// HTTP timeout (issue #6612: the fan-out previously ran with no deadline, so
// GET /v1/models hung for the full shared request timeout on every call).
func TestListAllModelsUnreachableProviderDoesNotBlockAggregate(t *testing.T) {
	healthySrv := httptest.NewServer(openAIStyleModelsHandler("gpt-4o"))
	t.Cleanup(healthySrv.Close)

	hang := make(chan struct{})
	hangingSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-hang
	}))
	// Release the hanging handler before closing the server so Close doesn't
	// wait on it. Registered before the client so Shutdown (registered later,
	// run earlier) still exercises the abandoned-call path first.
	t.Cleanup(func() {
		close(hang)
		hangingSrv.Close()
	})

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 10, healthySrv.URL)
	account.AddProviderWithBaseURL(schemas.Groq, 1, 10, hangingSrv.URL)
	account.SetKeysForProvider(schemas.OpenAI, wildcardKey(schemas.OpenAI))
	account.SetKeysForProvider(schemas.Groq, wildcardKey(schemas.Groq))
	tuneProviderForTest(t, account, schemas.OpenAI)
	tuneProviderForTest(t, account, schemas.Groq)

	// Shrink the per-provider poll deadline so the green path is fast; the
	// red path (no deadline honored) is still bounded by the 2s HTTP budget
	// above, which comfortably exceeds the assertion below.
	oldTimeout := listAllModelsProviderTimeout
	listAllModelsProviderTimeout = 400 * time.Millisecond
	t.Cleanup(func() { listAllModelsProviderTimeout = oldTimeout })

	client := newListModelsTestClient(t, account, nil)

	// No caller deadline, mirroring the HTTP handler: the aggregate itself
	// must bound the unreachable provider's poll.
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	defer ctx.Cancel()

	start := time.Now()
	resp, bfErr := client.ListAllModels(ctx, &schemas.BifrostListModelsRequest{})
	elapsed := time.Since(start)

	if bfErr != nil {
		t.Fatalf("ListAllModels returned error: %s", bfErr.GetErrorString())
	}
	if elapsed >= 1500*time.Millisecond {
		t.Fatalf("aggregate blocked for %v by the unreachable provider; want well under its 2s HTTP timeout", elapsed)
	}
	if !containsModelID(resp, "openai/gpt-4o") {
		t.Fatalf("healthy provider's models missing from aggregate; got %v", modelIDs(resp))
	}
}

// A direct key is the caller's own credential for one provider. Listing every provider would send it
// to each of them, so the aggregate refuses a request carrying one before any provider is asked,
// while a listing that names its provider still uses the key there, and only there.
func TestListAllModelsRefusesADirectKey(t *testing.T) {
	const directKey = "sk-caller-direct-key"
	var openAIAuth, groqAuth []string
	recording := func(seen *[]string) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			*seen = append(*seen, r.Header.Get("Authorization"))
			openAIStyleModelsHandler("model-a")(w, r)
		}
	}
	openAISrv := httptest.NewServer(recording(&openAIAuth))
	t.Cleanup(openAISrv.Close)
	groqSrv := httptest.NewServer(recording(&groqAuth))
	t.Cleanup(groqSrv.Close)

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 10, openAISrv.URL)
	account.AddProviderWithBaseURL(schemas.Groq, 1, 10, groqSrv.URL)
	account.SetKeysForProvider(schemas.OpenAI, wildcardKey(schemas.OpenAI))
	account.SetKeysForProvider(schemas.Groq, wildcardKey(schemas.Groq))
	tuneProviderForTest(t, account, schemas.OpenAI)
	tuneProviderForTest(t, account, schemas.Groq)
	client := newListModelsTestClient(t, account, nil)

	withDirectKey := func() *schemas.BifrostContext {
		ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
		ctx.SetValue(schemas.BifrostContextKeyDirectKey, schemas.Key{ID: "header-provided", Name: "header-provided", Value: *schemas.NewSecretVar(directKey), Weight: 1})
		return ctx
	}

	ctx := withDirectKey()
	defer ctx.Cancel()
	resp, bfErr := client.ListAllModels(ctx, &schemas.BifrostListModelsRequest{})
	if bfErr == nil {
		t.Fatalf("ListAllModels with a direct key returned %v; want a refusal naming the provider requirement (openai saw %v, groq saw %v)", modelIDs(resp), openAIAuth, groqAuth)
	}
	if bfErr.StatusCode == nil || *bfErr.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %v, want 400: %s", bfErr.StatusCode, bfErr.GetErrorString())
	}
	if len(openAIAuth) != 0 || len(groqAuth) != 0 {
		t.Fatalf("the direct key was sent to providers the caller never named: openai=%v groq=%v", openAIAuth, groqAuth)
	}

	named := withDirectKey()
	defer named.Cancel()
	if _, bfErr := client.ListModelsRequest(named, &schemas.BifrostListModelsRequest{Provider: schemas.OpenAI}); bfErr != nil {
		t.Fatalf("listing a named provider with a direct key failed: %s", bfErr.GetErrorString())
	}
	if len(openAIAuth) != 1 || openAIAuth[0] != "Bearer "+directKey {
		t.Fatalf("openai saw %v; want the direct key once", openAIAuth)
	}
	if len(groqAuth) != 0 {
		t.Fatalf("groq saw %v; a named listing must not reach another provider", groqAuth)
	}
}

// staticListerCatalog is a schemas.ModelInfoProvider that also implements the
// optional per-provider model listing the aggregate falls back on, mimicking
// the framework's ModelCatalog.
type staticListerCatalog struct {
	modelsByProvider map[schemas.ModelProvider][]string
}

func (c *staticListerCatalog) GetModelInfo(provider schemas.ModelProvider, model string) *schemas.Model {
	return nil
}

func (c *staticListerCatalog) CalculateRequestCost(ctx *schemas.BifrostContext, resp *schemas.BifrostResponse) float64 {
	return 0
}

func (c *staticListerCatalog) CalculateRequestCostBreakdown(ctx *schemas.BifrostContext, resp *schemas.BifrostResponse) *schemas.BifrostCost {
	return nil
}

func (c *staticListerCatalog) GetModelsForProvider(provider schemas.ModelProvider) []string {
	return c.modelsByProvider[provider]
}

func (c *staticListerCatalog) GetUnfilteredModelsForProvider(provider schemas.ModelProvider) []string {
	return c.modelsByProvider[provider]
}

// When a provider's live poll fails and a model catalog is configured, the
// aggregate should serve that provider's last-known models instead of
// dropping the provider from the response.
func TestListAllModelsServesLastKnownModelsWhenPollFails(t *testing.T) {
	healthySrv := httptest.NewServer(openAIStyleModelsHandler("gpt-4o"))
	t.Cleanup(healthySrv.Close)

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 10, healthySrv.URL)
	// Nothing listens here: the poll fails fast with connection refused.
	account.AddProviderWithBaseURL(schemas.Groq, 1, 10, "http://127.0.0.1:1")
	account.SetKeysForProvider(schemas.OpenAI, wildcardKey(schemas.OpenAI))
	account.SetKeysForProvider(schemas.Groq, wildcardKey(schemas.Groq))
	tuneProviderForTest(t, account, schemas.OpenAI)
	tuneProviderForTest(t, account, schemas.Groq)

	catalog := &staticListerCatalog{modelsByProvider: map[schemas.ModelProvider][]string{
		schemas.Groq: {"llama-3.3-70b-versatile"},
	}}
	client := newListModelsTestClient(t, account, catalog)

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	defer ctx.Cancel()

	resp, bfErr := client.ListAllModels(ctx, &schemas.BifrostListModelsRequest{})
	if bfErr != nil {
		t.Fatalf("ListAllModels returned error: %s", bfErr.GetErrorString())
	}
	if !containsModelID(resp, "openai/gpt-4o") {
		t.Fatalf("healthy provider's models missing from aggregate; got %v", modelIDs(resp))
	}
	if !containsModelID(resp, "groq/llama-3.3-70b-versatile") {
		t.Fatalf("last-known models for the failed provider missing from aggregate; got %v", modelIDs(resp))
	}
}

// With no catalog to fall back on, an aggregate where every poll failed must
// keep returning the first error rather than an empty success.
func TestListAllModelsAllPollsFailWithoutCatalogReturnsError(t *testing.T) {
	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.Groq, 1, 10, "http://127.0.0.1:1")
	account.SetKeysForProvider(schemas.Groq, wildcardKey(schemas.Groq))
	tuneProviderForTest(t, account, schemas.Groq)

	client := newListModelsTestClient(t, account, nil)

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	defer ctx.Cancel()

	resp, bfErr := client.ListAllModels(ctx, &schemas.BifrostListModelsRequest{})
	if bfErr == nil {
		t.Fatalf("expected an error when every provider poll fails with no catalog fallback; got %v", modelIDs(resp))
	}
}

// modelDenyPlugin drops every listed model whose bare name is in deny, in
// PostLLMHook, the way the governance plugin narrows a listing to what a
// virtual key may see.
type modelDenyPlugin struct {
	deny map[string]bool
}

func (p *modelDenyPlugin) GetName() string { return "model-deny" }
func (p *modelDenyPlugin) Cleanup() error  { return nil }
func (p *modelDenyPlugin) PreRequestHook(ctx *schemas.BifrostContext, req *schemas.BifrostRequest) error {
	return nil
}
func (p *modelDenyPlugin) PreLLMHook(ctx *schemas.BifrostContext, req *schemas.BifrostRequest) (*schemas.BifrostRequest, *schemas.LLMPluginShortCircuit, error) {
	return req, nil, nil
}
func (p *modelDenyPlugin) PostLLMHook(ctx *schemas.BifrostContext, resp *schemas.BifrostResponse, bifrostErr *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError, error) {
	if resp == nil || resp.ListModelsResponse == nil {
		return resp, bifrostErr, nil
	}
	kept := make([]schemas.Model, 0, len(resp.ListModelsResponse.Data))
	for _, m := range resp.ListModelsResponse.Data {
		if _, name := schemas.ParseModelString(m.ID, ""); !p.deny[name] {
			kept = append(kept, m)
		}
	}
	resp.ListModelsResponse.Data = kept
	return resp, bifrostErr, nil
}

// The last-known models served for a failed poll must pass through the plugin
// post-hooks like a live listing does; otherwise a virtual key's model
// allow/block lists are skipped whenever its provider is unreachable.
func TestListAllModelsLastKnownModelsPassThroughPostHooks(t *testing.T) {
	account := NewMockAccount()
	// Nothing listens here: the poll fails fast with connection refused.
	account.AddProviderWithBaseURL(schemas.Groq, 1, 10, "http://127.0.0.1:1")
	account.SetKeysForProvider(schemas.Groq, wildcardKey(schemas.Groq))
	tuneProviderForTest(t, account, schemas.Groq)

	catalog := &staticListerCatalog{modelsByProvider: map[schemas.ModelProvider][]string{
		schemas.Groq: {"llama-3.3-70b-versatile", "llama-guard-4"},
	}}
	client, initErr := Init(context.Background(), schemas.BifrostConfig{
		Account:      account,
		Logger:       NewNoOpLogger(),
		ModelCatalog: catalog,
		LLMPlugins:   []schemas.LLMPlugin{&modelDenyPlugin{deny: map[string]bool{"llama-guard-4": true}}},
	})
	if initErr != nil {
		t.Fatalf("Init failed: %v", initErr)
	}
	t.Cleanup(client.Shutdown)

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	defer ctx.Cancel()

	resp, bfErr := client.ListAllModels(ctx, &schemas.BifrostListModelsRequest{})
	if bfErr != nil {
		t.Fatalf("ListAllModels returned error: %s", bfErr.GetErrorString())
	}
	if !containsModelID(resp, "groq/llama-3.3-70b-versatile") {
		t.Fatalf("last-known model the plugin allows is missing; got %v", modelIDs(resp))
	}
	if containsModelID(resp, "groq/llama-guard-4") {
		t.Fatalf("last-known model the plugin's PostLLMHook drops leaked into the aggregate; got %v", modelIDs(resp))
	}
}

// A single-provider list call keeps returning the poll error: the last-known
// fallback belongs to the aggregate only.
func TestListModelsRequestDoesNotServeLastKnownModels(t *testing.T) {
	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.Groq, 1, 10, "http://127.0.0.1:1")
	account.SetKeysForProvider(schemas.Groq, wildcardKey(schemas.Groq))
	tuneProviderForTest(t, account, schemas.Groq)

	catalog := &staticListerCatalog{modelsByProvider: map[schemas.ModelProvider][]string{
		schemas.Groq: {"llama-3.3-70b-versatile"},
	}}
	client := newListModelsTestClient(t, account, catalog)

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	defer ctx.Cancel()

	resp, bfErr := client.ListModelsRequest(ctx, &schemas.BifrostListModelsRequest{Provider: schemas.Groq})
	if bfErr == nil {
		t.Fatalf("expected the poll error from a single-provider list call; got %v", modelIDs(resp))
	}
}

// BIT-2567: a caller that set a deadline gets the whole of it for each
// provider's poll, as in v1.11.3; listAllModelsProviderTimeout bounds a poll
// only when the caller set none. Durations are scaled down: the provider
// answers after slowAnswer, longer than the per-provider bound and well inside
// the caller's deadline (bitrouter: a 7 s vendor under a 15 s caller budget
// against the 5 s bound).
func TestListAllModelsHonoursCallerDeadline(t *testing.T) {
	const slowAnswer = 600 * time.Millisecond
	release := make(chan struct{})
	slowSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-time.After(slowAnswer):
		case <-release:
			return
		}
		openAIStyleModelsHandler("gpt-4o")(w, r)
	}))
	t.Cleanup(func() {
		close(release)
		slowSrv.Close()
	})

	account := NewMockAccount()
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 10, slowSrv.URL)
	account.SetKeysForProvider(schemas.OpenAI, wildcardKey(schemas.OpenAI))
	tuneProviderForTest(t, account, schemas.OpenAI)

	oldTimeout := listAllModelsProviderTimeout
	listAllModelsProviderTimeout = 300 * time.Millisecond
	t.Cleanup(func() { listAllModelsProviderTimeout = oldTimeout })

	client := newListModelsTestClient(t, account, nil)

	t.Run("caller deadline outlasts the slow answer", func(t *testing.T) {
		ctx := schemas.NewBifrostContext(context.Background(), time.Now().Add(1500*time.Millisecond))
		defer ctx.Cancel()
		resp, bfErr := client.ListAllModels(ctx, &schemas.BifrostListModelsRequest{})
		if bfErr != nil {
			t.Fatalf("ListAllModels returned error: %s", bfErr.GetErrorString())
		}
		if !containsModelID(resp, "openai/gpt-4o") {
			t.Fatalf("a %v answer inside the caller's deadline was cut by the per-provider bound; got %v", slowAnswer, modelIDs(resp))
		}
	})

	t.Run("no caller deadline keeps the per-provider bound", func(t *testing.T) {
		ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
		defer ctx.Cancel()
		start := time.Now()
		resp, _ := client.ListAllModels(ctx, &schemas.BifrostListModelsRequest{})
		elapsed := time.Since(start)
		if containsModelID(resp, "openai/gpt-4o") {
			t.Fatalf("a %v answer was served past the %v per-provider bound", slowAnswer, listAllModelsProviderTimeout)
		}
		if elapsed >= slowAnswer {
			t.Fatalf("aggregate waited %v; want the %v per-provider bound to end the poll first", elapsed, listAllModelsProviderTimeout)
		}
	})
}

package bifrost

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	schemas "github.com/maximhq/bifrost/core/schemas"
)

// Typesafe as a custom-provider base (BIT-2398): an org can run several
// TypeSafe instances, each a custom provider based on typesafe with its own
// BaseURL and keys.

const typesafeOnPrem schemas.ModelProvider = "typesafe-onprem"

// typesafeServer records every hit (path and Authorization header) and answers
// a systemone request with a valid noul answer, or with status when non-zero.
type typesafeServer struct {
	*httptest.Server
	hits   atomic.Int32
	mu     sync.Mutex
	paths  []string
	auths  []string
	models []string
}

func newTypesafeServer(t *testing.T, status int) *typesafeServer {
	t.Helper()
	s := &typesafeServer{}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.hits.Add(1)
		var body struct {
			Model string `json:"model"`
		}
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		s.mu.Lock()
		s.paths = append(s.paths, r.URL.Path)
		s.auths = append(s.auths, r.Header.Get("Authorization"))
		s.models = append(s.models, body.Model)
		s.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if status != 0 {
			w.WriteHeader(status)
			_, _ = w.Write([]byte(`{"detail":"upstream rejected the request"}`))
			return
		}
		_, _ = w.Write([]byte(`{"model":"jev-1.13.0","answers":{"approve":{"type":"noul","noul":0.9,"confidence":0.8}},"usage":{"input_tokens":12,"output_tokens":0}}`))
	}))
	t.Cleanup(s.Close)
	return s
}

func (s *typesafeServer) snapshot() (paths, auths, models []string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.paths...), append([]string(nil), s.auths...), append([]string(nil), s.models...)
}

// addTypesafeCustomProvider configures name as a custom provider based on
// typesafe, pointed at baseURL, with one key.
func addTypesafeCustomProvider(account *MockAccount, name schemas.ModelProvider, baseURL, keyValue string, cpc *schemas.CustomProviderConfig) {
	account.AddProviderWithBaseURL(name, 1, 1, baseURL)
	account.configs[name].NetworkConfig.MaxRetries = 0
	if cpc == nil {
		cpc = &schemas.CustomProviderConfig{}
	}
	cpc.BaseProviderType = schemas.Typesafe
	account.SetCustomProviderConfig(name, cpc)
	account.SetKeysForProvider(name, []schemas.Key{{
		ID: string(name) + "-key", Value: *schemas.NewSecretVar(keyValue),
		Models: schemas.WhiteList{"*"}, Weight: 100,
	}})
}

func noulDecisionRequest(provider schemas.ModelProvider) *schemas.BifrostDecisionRequest {
	return &schemas.BifrostDecisionRequest{
		Provider: provider,
		Model:    "jev-1.13.0",
		State:    "the user asked for a refund",
		Questions: map[string]schemas.DecisionQuestion{
			"approve": {Kind: schemas.DecisionKindNoul, Instructions: "Approve the refund?"},
		},
	}
}

func decisionTestContext() *schemas.BifrostContext {
	return schemas.NewBifrostContext(context.Background(), time.Now().Add(10*time.Second))
}

func TestTypesafeIsSupportedBaseProvider(t *testing.T) {
	if !IsSupportedBaseProvider(schemas.Typesafe) {
		t.Fatal("typesafe must be accepted as a custom-provider base")
	}
}

func TestTypesafeCustomProviderDecisionReachesOwnBaseURL(t *testing.T) {
	server := newTypesafeServer(t, 0)
	account := NewMockAccount()
	addTypesafeCustomProvider(account, typesafeOnPrem, server.URL, "ts-onprem-secret", nil)
	client := newStreamTestClient(t, account)

	resp, bifrostErr := client.DecisionRequest(decisionTestContext(), noulDecisionRequest(typesafeOnPrem))
	if bifrostErr != nil {
		t.Fatalf("decision failed: %+v", bifrostErr.Error)
	}
	if got := server.hits.Load(); got != 1 {
		t.Fatalf("custom typesafe server hits = %d, want 1", got)
	}
	paths, auths, models := server.snapshot()
	if paths[0] != "/v1/systemone" {
		t.Errorf("path = %q, want /v1/systemone", paths[0])
	}
	if auths[0] != "Bearer ts-onprem-secret" {
		t.Errorf("Authorization = %q, want the custom provider's own key", auths[0])
	}
	if models[0] != "jev-1.13.0" {
		t.Errorf("upstream model = %q, want bare jev-1.13.0", models[0])
	}
	if resp.ExtraFields.Provider != typesafeOnPrem {
		t.Errorf("response ExtraFields.Provider = %q, want %q", resp.ExtraFields.Provider, typesafeOnPrem)
	}
	if got := resp.Answers["approve"].Value; got != 0.9 {
		t.Errorf("approve answer = %v, want 0.9", got)
	}
}

func TestTypesafeCustomProviderDecisionErrorReportsCustomProvider(t *testing.T) {
	server := newTypesafeServer(t, http.StatusUnprocessableEntity)
	account := NewMockAccount()
	addTypesafeCustomProvider(account, typesafeOnPrem, server.URL, "ts-onprem-secret", nil)
	client := newStreamTestClient(t, account)

	_, bifrostErr := client.DecisionRequest(decisionTestContext(), noulDecisionRequest(typesafeOnPrem))
	if bifrostErr == nil {
		t.Fatal("expected the upstream 422 to surface as an error")
	}
	if bifrostErr.ExtraFields.Provider != typesafeOnPrem {
		t.Errorf("error ExtraFields.Provider = %q, want %q", bifrostErr.ExtraFields.Provider, typesafeOnPrem)
	}
	if bifrostErr.StatusCode == nil || *bifrostErr.StatusCode != http.StatusUnprocessableEntity {
		t.Errorf("status = %v, want 422", bifrostErr.StatusCode)
	}
	if got := server.hits.Load(); got != 1 {
		t.Errorf("server hits = %d, want 1", got)
	}
}

func TestTwoTypesafeCustomProvidersReachTheirOwnServers(t *testing.T) {
	serverA := newTypesafeServer(t, 0)
	serverB := newTypesafeServer(t, 0)
	const providerA, providerB schemas.ModelProvider = "typesafe-eu", "typesafe-us"
	account := NewMockAccount()
	addTypesafeCustomProvider(account, providerA, serverA.URL, "ts-eu-secret", nil)
	addTypesafeCustomProvider(account, providerB, serverB.URL, "ts-us-secret", nil)
	client := newStreamTestClient(t, account)

	for _, tc := range []struct {
		provider schemas.ModelProvider
		server   *typesafeServer
		other    *typesafeServer
		auth     string
	}{
		{providerA, serverA, serverB, "Bearer ts-eu-secret"},
		{providerB, serverB, serverA, "Bearer ts-us-secret"},
	} {
		before := tc.other.hits.Load()
		resp, bifrostErr := client.DecisionRequest(decisionTestContext(), noulDecisionRequest(tc.provider))
		if bifrostErr != nil {
			t.Fatalf("%s: decision failed: %+v", tc.provider, bifrostErr.Error)
		}
		if resp.ExtraFields.Provider != tc.provider {
			t.Errorf("%s: ExtraFields.Provider = %q", tc.provider, resp.ExtraFields.Provider)
		}
		if got := tc.other.hits.Load(); got != before {
			t.Errorf("%s: request leaked to the other instance's server", tc.provider)
		}
		_, auths, _ := tc.server.snapshot()
		if len(auths) != 1 || auths[0] != tc.auth {
			t.Errorf("%s: Authorization headers = %v, want [%q]", tc.provider, auths, tc.auth)
		}
	}
}

func TestTypesafeCustomProviderAllowedRequestsDenyDecision(t *testing.T) {
	server := newTypesafeServer(t, 0)
	account := NewMockAccount()
	addTypesafeCustomProvider(account, typesafeOnPrem, server.URL, "ts-onprem-secret", &schemas.CustomProviderConfig{
		AllowedRequests: &schemas.AllowedRequests{ListModels: true},
	})
	client := newStreamTestClient(t, account)

	_, bifrostErr := client.DecisionRequest(decisionTestContext(), noulDecisionRequest(typesafeOnPrem))
	if !isUnsupportedOperation(bifrostErr) {
		t.Fatalf("expected unsupported_operation, got %+v", bifrostErr)
	}
	if got := server.hits.Load(); got != 0 {
		t.Errorf("a denied decision reached the upstream %d time(s)", got)
	}
}

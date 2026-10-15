package typesafe

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	schemas "github.com/maximhq/bifrost/core/schemas"
)

// Typesafe as a custom-provider base (BIT-2398), ported to the v1.12.0
// decision schema (BIT-2567).

const onPremKey schemas.ModelProvider = "typesafe-onprem"

func onPremConfig(baseURL string, mutate func(*schemas.CustomProviderConfig)) *schemas.ProviderConfig {
	cpc := &schemas.CustomProviderConfig{
		CustomProviderKey: string(onPremKey),
		BaseProviderType:  schemas.Typesafe,
	}
	if mutate != nil {
		mutate(cpc)
	}
	return &schemas.ProviderConfig{
		NetworkConfig:        schemas.NetworkConfig{BaseURL: baseURL},
		CustomProviderConfig: cpc,
	}
}

func newTestProvider(t *testing.T, config *schemas.ProviderConfig) *TypesafeProvider {
	t.Helper()
	provider, err := NewTypesafeProvider(config, noopLogger{})
	if err != nil {
		t.Fatalf("NewTypesafeProvider: %v", err)
	}
	return provider
}

// pathRecorder answers every request with a valid single-noul systemone
// response and records the request paths.
type pathRecorder struct {
	mu    sync.Mutex
	paths []string
}

func (p *pathRecorder) server(t *testing.T) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p.mu.Lock()
		p.paths = append(p.paths, r.URL.Path)
		p.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"jev-1.13.0","answers":{"approve":{"type":"noul","noul":0.9,"confidence":0.8}}}`))
	}))
	t.Cleanup(server.Close)
	return server
}

func (p *pathRecorder) recorded() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]string(nil), p.paths...)
}

// noulRequest is one predicate ("noul") question named approve over a text state.
func noulRequest() *schemas.BifrostDecisionRequest {
	return &schemas.BifrostDecisionRequest{
		Provider: schemas.Typesafe,
		Model:    "jev-1.13.0",
		Input:    schemas.DecisionInput{Text: schemas.Ptr("state")},
		Questions: []schemas.DecisionQuestion{
			{Type: schemas.DecisionTypePredicate, Name: schemas.Ptr("approve"), Instructions: schemas.NewDecisionText("Approve?")},
		},
	}
}

func testContext() *schemas.BifrostContext {
	return schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
}

func TestGetProviderKey(t *testing.T) {
	native := newTestProvider(t, &schemas.ProviderConfig{})
	if got := native.GetProviderKey(); got != schemas.Typesafe {
		t.Errorf("native GetProviderKey = %q, want typesafe", got)
	}
	custom := newTestProvider(t, onPremConfig("", nil))
	if got := custom.GetProviderKey(); got != onPremKey {
		t.Errorf("custom GetProviderKey = %q, want %q", got, onPremKey)
	}
}

func TestDecisionRequestPath(t *testing.T) {
	tests := []struct {
		name     string
		config   func(baseURL string) *schemas.ProviderConfig
		ctxPath  string
		wantPath string
	}{
		{
			name: "native default path",
			config: func(baseURL string) *schemas.ProviderConfig {
				return &schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{BaseURL: baseURL}}
			},
			wantPath: "/v1/systemone",
		},
		{
			name: "native context path override unchanged",
			config: func(baseURL string) *schemas.ProviderConfig {
				return &schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{BaseURL: baseURL}}
			},
			ctxPath:  "/v1/systemone-ctx",
			wantPath: "/v1/systemone-ctx",
		},
		{
			name:     "custom provider default path",
			config:   func(baseURL string) *schemas.ProviderConfig { return onPremConfig(baseURL, nil) },
			wantPath: "/v1/systemone",
		},
		{
			name: "custom provider RequestPathOverrides",
			config: func(baseURL string) *schemas.ProviderConfig {
				return onPremConfig(baseURL, func(cpc *schemas.CustomProviderConfig) {
					cpc.RequestPathOverrides = map[schemas.RequestType]string{schemas.DecisionRequest: "jev/evaluate"}
				})
			},
			wantPath: "/jev/evaluate",
		},
		{
			name: "override for another request type is ignored",
			config: func(baseURL string) *schemas.ProviderConfig {
				return onPremConfig(baseURL, func(cpc *schemas.CustomProviderConfig) {
					cpc.RequestPathOverrides = map[schemas.RequestType]string{schemas.ChatCompletionRequest: "/chat"}
				})
			},
			wantPath: "/v1/systemone",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			recorder := &pathRecorder{}
			server := recorder.server(t)
			provider := newTestProvider(t, tc.config(server.URL))
			ctx := testContext()
			if tc.ctxPath != "" {
				ctx.SetValue(schemas.BifrostContextKeyURLPath, tc.ctxPath)
			}
			if _, bifrostErr := provider.Decision(ctx, schemas.Key{}, noulRequest()); bifrostErr != nil {
				t.Fatalf("Decision: %+v", bifrostErr.Error)
			}
			if got := recorder.recorded(); len(got) != 1 || got[0] != tc.wantPath {
				t.Fatalf("paths = %v, want [%q]", got, tc.wantPath)
			}
		})
	}
}

func TestDecisionAbsoluteURLOverride(t *testing.T) {
	recorder := &pathRecorder{}
	server := recorder.server(t)
	provider := newTestProvider(t, onPremConfig("https://unused.invalid", func(cpc *schemas.CustomProviderConfig) {
		cpc.RequestPathOverrides = map[schemas.RequestType]string{schemas.DecisionRequest: server.URL + "/abs/systemone"}
	}))
	if _, bifrostErr := provider.Decision(testContext(), schemas.Key{}, noulRequest()); bifrostErr != nil {
		t.Fatalf("Decision: %+v", bifrostErr.Error)
	}
	if got := recorder.recorded(); len(got) != 1 || got[0] != "/abs/systemone" {
		t.Fatalf("paths = %v, want [/abs/systemone]", got)
	}
}

func TestAllowedRequestsGateOperations(t *testing.T) {
	recorder := &pathRecorder{}
	server := recorder.server(t)
	denyAll := newTestProvider(t, onPremConfig(server.URL, func(cpc *schemas.CustomProviderConfig) {
		cpc.AllowedRequests = &schemas.AllowedRequests{}
	}))

	_, bifrostErr := denyAll.Decision(testContext(), schemas.Key{}, noulRequest())
	assertUnsupported(t, "Decision", bifrostErr)
	_, bifrostErr = denyAll.ListModels(testContext(), []schemas.Key{{Models: schemas.WhiteList{"*"}}}, &schemas.BifrostListModelsRequest{Provider: onPremKey})
	assertUnsupported(t, "ListModels", bifrostErr)
	if got := recorder.recorded(); len(got) != 0 {
		t.Fatalf("a denied operation reached the upstream: %v", got)
	}

	allowDecision := newTestProvider(t, onPremConfig(server.URL, func(cpc *schemas.CustomProviderConfig) {
		cpc.AllowedRequests = &schemas.AllowedRequests{Decision: true}
	}))
	if _, bifrostErr := allowDecision.Decision(testContext(), schemas.Key{}, noulRequest()); bifrostErr != nil {
		t.Fatalf("allowed Decision failed: %+v", bifrostErr.Error)
	}
}

func assertUnsupported(t *testing.T, op string, bifrostErr *schemas.BifrostError) {
	t.Helper()
	if bifrostErr == nil || bifrostErr.Error == nil || bifrostErr.Error.Code == nil || *bifrostErr.Error.Code != "unsupported_operation" {
		t.Fatalf("%s: expected unsupported_operation, got %+v", op, bifrostErr)
	}
	if bifrostErr.ExtraFields.Provider != onPremKey {
		t.Errorf("%s: error provider = %q, want %q", op, bifrostErr.ExtraFields.Provider, onPremKey)
	}
}

// TestListModelsUsesProviderKey: with no base URL configured, the native and
// the custom provider both list the pinned jev catalog under their own key.
// v1.12.0 also asks the endpoint's live GET /v1/models; the default endpoint
// is redirected to a loopback stub that fails, so the pinned catalog is what
// is served (as it is when api.typesafe.ai is unreachable) and the test never
// reaches the network.
func TestListModelsUsesProviderKey(t *testing.T) {
	keys := []schemas.Key{{Models: schemas.WhiteList{"*"}}}
	unavailable := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	t.Cleanup(unavailable.Close)
	for _, tc := range []struct {
		name   string
		config *schemas.ProviderConfig
		prefix string
	}{
		{"native", &schemas.ProviderConfig{}, "typesafe/"},
		{"custom", onPremConfig("", nil), "typesafe-onprem/"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			provider := newTestProvider(t, tc.config)
			provider.networkConfig.BaseURL = unavailable.URL
			resp, bifrostErr := provider.ListModels(testContext(), keys, &schemas.BifrostListModelsRequest{Provider: provider.GetProviderKey()})
			if bifrostErr != nil {
				t.Fatalf("ListModels: %+v", bifrostErr.Error)
			}
			if len(resp.Data) != len(typesafeModels) {
				t.Fatalf("got %d models, want %d", len(resp.Data), len(typesafeModels))
			}
			for i, model := range resp.Data {
				if want := tc.prefix + typesafeModels[i].ID; model.ID != want {
					t.Errorf("model[%d].ID = %q, want %q", i, model.ID, want)
				}
			}
		})
	}
}

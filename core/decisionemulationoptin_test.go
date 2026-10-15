package bifrost

import (
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	schemas "github.com/maximhq/bifrost/core/schemas"
)

// BIT-2398's decision-emulation tests, ported to v1.12.0 (BIT-2567).
//
// The fork's opt-out key (BifrostContextKeyDisableDecisionEmulation) is gone:
// in v1.12.0 core never emulates a decision unless the request is marked for
// it (BifrostContextKeyChangeRequestType = ResponsesRequest), and the only
// marker is the compat plugin's convert_decision_to_responses, switched on per
// request by BifrostContextKeyCompatConvertDecisionToResponses. So the
// opt-out's safety property now holds with no key at all: an unmarked decision
// whose primary and fallback providers answer unsupported_operation fails with
// that error and never reaches a Responses endpoint. Emulation is the opt-in.

// isDecisionUnsupportedOperation reports whether err carries the
// unsupported_operation code.
func isDecisionUnsupportedOperation(err *schemas.BifrostError) bool {
	return err != nil && err.Error != nil && err.Error.Code != nil && *err.Error.Code == "unsupported_operation"
}

// openAIResponsesServer counts every hit and answers /v1/responses with an
// emit_decision function call for the single "approve" noul question.
func openAIResponsesServer(t *testing.T) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		if r.URL.Path != "/v1/responses" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","object":"response","created_at":1700000000,"model":"gpt-4o-mini","status":"completed",` +
			`"output":[{"type":"function_call","id":"fc_1","call_id":"call_1","name":"emit_decision","status":"completed",` +
			`"arguments":"{\"approve\":{\"value\":0.9,\"confidence\":0.8}}"}],` +
			`"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}`))
	}))
	t.Cleanup(server.Close)
	return server, &hits
}

func addOpenAIProvider(account *MockAccount, baseURL string) {
	account.AddProviderWithBaseURL(schemas.OpenAI, 1, 1, baseURL)
	account.configs[schemas.OpenAI].NetworkConfig.MaxRetries = 0
	account.SetKeysForProvider(schemas.OpenAI, []schemas.Key{{
		ID: "openai-key", Value: *schemas.NewSecretVar("sk-openai"),
		Models: schemas.WhiteList{"*"}, Weight: 100,
	}})
}

func openAIDecisionRequest() *schemas.BifrostDecisionRequest {
	req := noulDecisionRequest(schemas.OpenAI)
	req.Model = "gpt-4o-mini"
	return req
}

// compatDecisionMarker stands in for the compat plugin's
// convert_decision_to_responses (plugins/compat lives outside the core module):
// on every attempt, a decision whose context carries
// BifrostContextKeyCompatConvertDecisionToResponses=true and whose model is not
// a native decisions model is marked for emulation, as markDecisionForEmulation
// does. core clears the mark between attempts (clearCtxForFallback), and the
// plugin re-marks each one.
type compatDecisionMarker struct{}

func (compatDecisionMarker) GetName() string { return "compat-decision-marker" }
func (compatDecisionMarker) Cleanup() error  { return nil }
func (compatDecisionMarker) PreRequestHook(*schemas.BifrostContext, *schemas.BifrostRequest) error {
	return nil
}

func (compatDecisionMarker) PreLLMHook(ctx *schemas.BifrostContext, req *schemas.BifrostRequest) (*schemas.BifrostRequest, *schemas.LLMPluginShortCircuit, error) {
	if enabled, _ := ctx.Value(schemas.BifrostContextKeyCompatConvertDecisionToResponses).(bool); !enabled {
		return req, nil, nil
	}
	if req.RequestType != schemas.DecisionRequest || req.DecisionRequest == nil {
		return req, nil, nil
	}
	model := req.DecisionRequest.Model
	if schemas.IsTypesafeModel(model) || schemas.DefaultSupportsDecisions(model) {
		return req, nil, nil
	}
	ctx.SetValue(schemas.BifrostContextKeyChangeRequestType, schemas.ResponsesRequest)
	return req, nil, nil
}

func (compatDecisionMarker) PostLLMHook(_ *schemas.BifrostContext, resp *schemas.BifrostResponse, bifrostErr *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError, error) {
	return resp, bifrostErr, nil
}

// Port of TestDecisionEmulationOptOutReturnsUnsupportedOperation: without any
// key, a provider without native decision support answers
// unsupported_operation and the decision is never turned into a Responses call.
func TestDecisionWithoutCompatMarkReturnsUnsupportedOperation(t *testing.T) {
	server, hits := openAIResponsesServer(t)
	account := NewMockAccount()
	addOpenAIProvider(account, server.URL)
	client := newStreamTestClient(t, account)

	_, bifrostErr := client.DecisionRequest(decisionTestContext(), openAIDecisionRequest())
	if !isDecisionUnsupportedOperation(bifrostErr) {
		t.Fatalf("expected unsupported_operation, got %+v", bifrostErr)
	}
	if got := hits.Load(); got != 0 {
		t.Fatalf("unmarked decision reached the OpenAI server %d time(s), want 0", got)
	}
}

// Port of TestDecisionEmulationOptOutHoldsOnFallback: without any key, the
// property holds on a fallback too.
func TestDecisionWithoutCompatMarkHoldsOnFallback(t *testing.T) {
	primary := newTypesafeServer(t, http.StatusInternalServerError)
	server, hits := openAIResponsesServer(t)
	account := NewMockAccount()
	addTypesafeCustomProvider(account, typesafeOnPrem, primary.URL, "ts-onprem-secret", nil)
	addOpenAIProvider(account, server.URL)
	client := newStreamTestClient(t, account)

	ctx := decisionTestContext()
	req := noulDecisionRequest(typesafeOnPrem)
	req.Fallbacks = []schemas.Fallback{{Provider: schemas.OpenAI, Model: "gpt-4o-mini"}}
	_, bifrostErr := client.DecisionRequest(ctx, req)
	// Core returns the primary's error once every fallback has failed; the
	// unmarked fallback failed with unsupported_operation instead of emulating.
	if bifrostErr == nil || bifrostErr.ExtraFields.Provider != typesafeOnPrem {
		t.Fatalf("expected the primary's error after the fallback failed, got %+v", bifrostErr)
	}
	if got := primary.hits.Load(); got != 1 {
		t.Fatalf("primary hits = %d, want 1 (the fallback must have been tried after it)", got)
	}
	if got := hits.Load(); got != 0 {
		t.Fatalf("unmarked fallback reached the OpenAI server %d time(s), want 0", got)
	}
	if changeType, ok := ctx.Value(schemas.BifrostContextKeyChangeRequestType).(schemas.RequestType); ok && changeType == schemas.ResponsesRequest {
		t.Error("no attempt may leave the decision marked for emulation")
	}
}

// Port of TestDecisionEmulationWithoutOptOutStillEmulates: the compat opt-in
// (BifrostContextKeyCompatConvertDecisionToResponses, marked per attempt by
// the compat plugin) turns emulation through the Responses API on, on the
// primary and on a fallback.
func TestDecisionCompatOptInEmulates(t *testing.T) {
	optedIn := func() *schemas.BifrostContext {
		ctx := decisionTestContext()
		ctx.SetValue(schemas.BifrostContextKeyCompatConvertDecisionToResponses, true)
		return ctx
	}
	t.Run("primary", func(t *testing.T) {
		server, hits := openAIResponsesServer(t)
		account := NewMockAccount()
		addOpenAIProvider(account, server.URL)
		client := newStreamTestClient(t, account, compatDecisionMarker{})

		resp, bifrostErr := client.DecisionRequest(optedIn(), openAIDecisionRequest())
		if bifrostErr != nil {
			t.Fatalf("emulated decision failed: %+v", bifrostErr.Error)
		}
		if got := hits.Load(); got != 1 {
			t.Fatalf("OpenAI server hits = %d, want 1", got)
		}
		if got := decisionAnswerProbability(t, resp, "approve"); got != 0.9 {
			t.Errorf("approve answer = %v, want 0.9", got)
		}
	})
	t.Run("fallback", func(t *testing.T) {
		primary := newTypesafeServer(t, http.StatusInternalServerError)
		server, hits := openAIResponsesServer(t)
		account := NewMockAccount()
		addTypesafeCustomProvider(account, typesafeOnPrem, primary.URL, "ts-onprem-secret", nil)
		addOpenAIProvider(account, server.URL)
		client := newStreamTestClient(t, account, compatDecisionMarker{})

		req := noulDecisionRequest(typesafeOnPrem)
		req.Fallbacks = []schemas.Fallback{{Provider: schemas.OpenAI, Model: "gpt-4o-mini"}}
		resp, bifrostErr := client.DecisionRequest(optedIn(), req)
		if bifrostErr != nil {
			t.Fatalf("emulated fallback decision failed: %+v", bifrostErr.Error)
		}
		if got := hits.Load(); got != 1 {
			t.Fatalf("OpenAI server hits = %d, want 1", got)
		}
		if resp.ExtraFields.Provider != schemas.OpenAI {
			t.Errorf("ExtraFields.Provider = %q, want openai", resp.ExtraFields.Provider)
		}
	})
}

// Port of TestDecisionEmulationOptOutSurvivesContextClears. The opt-out key no
// longer exists, so what must survive the per-fallback and internal-request
// clears is its replacement, the absence of the emulation mark: neither clear
// may mark an unmarked decision context (fail closed). The opt-out's own
// nil-context and non-bool-value assertions have no v1.12.0 counterpart.
func TestDecisionUnmarkedContextStaysUnmarkedAcrossClears(t *testing.T) {
	isMarked := func(ctx *schemas.BifrostContext) bool {
		changeType, ok := ctx.Value(schemas.BifrostContextKeyChangeRequestType).(schemas.RequestType)
		return ok && changeType == schemas.ResponsesRequest
	}
	ctx := decisionTestContext()
	clearCtxForFallback(ctx)
	if isMarked(ctx) {
		t.Fatal("clearCtxForFallback marked an unmarked decision for emulation")
	}
	ClearContextForInternalRequest(ctx)
	if isMarked(ctx) {
		t.Fatal("ClearContextForInternalRequest marked an unmarked decision for emulation")
	}
	// A mark left by one attempt does not carry into the next: the fallback
	// is emulated only if the compat plugin marks it again.
	ctx.SetValue(schemas.BifrostContextKeyChangeRequestType, schemas.ResponsesRequest)
	clearCtxForFallback(ctx)
	if isMarked(ctx) {
		t.Fatal("clearCtxForFallback carried the emulation mark into the next attempt")
	}
}

package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"testing"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const refusalTestModel = "claude-sonnet-4-5"

var refusalTestProviders = []schemas.ModelProvider{schemas.Anthropic, schemas.Azure, schemas.Vertex}

// refusalRow is one row of the BIT-2954 § Refusals table: how a request carries the feature, and
// the ProviderFeatures gate that, turned off, makes the target unable to carry it.
type refusalRow struct {
	feature Feature
	name    string
	// params is the chat request's parameters as JSON (typed fields).
	params string
	// extra are ExtraParams entries the converter promotes onto typed fields.
	extra map[string]any
	// raw are ExtraParams entries as json.RawMessage, merged onto the body under the
	// passthrough flag (an unconsumed top-level key), when the feature has a top-level spelling.
	raw     map[string]string
	disable func(*ProviderFeatureSupport)
	ctx     func(*schemas.BifrostContext)
}

func refusalRows() []refusalRow {
	return []refusalRow{
		{feature: FeatureContainer, name: "container", params: `{"container":"c1"}`, raw: map[string]string{"container": `"c1"`},
			disable: func(f *ProviderFeatureSupport) { f.ContainerBasic = false }},
		{feature: FeatureContainerSkills, name: "container_skills", params: `{"container":{"id":"c1","skills":[{"type":"anthropic","skill_id":"pdf"}]}}`,
			raw:     map[string]string{"container": `{"id":"c1","skills":[{"type":"anthropic","skill_id":"pdf"}]}`},
			disable: func(f *ProviderFeatureSupport) { f.Skills = false }},
		{feature: FeatureMCPServers, name: "mcp_servers", params: `{"mcp_servers":[{"type":"url","url":"https://mcp.example","name":"m"}]}`,
			raw:     map[string]string{"mcp_servers": `[{"type":"url","url":"https://mcp.example","name":"m"}]`},
			disable: func(f *ProviderFeatureSupport) { f.MCP = false }},
		{feature: FeatureSpeed, name: "speed", params: `{"speed":"fast"}`, raw: map[string]string{"speed": `"fast"`},
			disable: func(f *ProviderFeatureSupport) { f.FastMode = false }},
		{feature: FeatureTaskBudget, name: "task_budget", params: `{"task_budget":{"type":"tokens","total":1000}}`,
			disable: func(f *ProviderFeatureSupport) { f.TaskBudgets = false }},
		{feature: FeatureInferenceGeo, name: "inference_geo", params: `{"inference_geo":"us"}`, raw: map[string]string{"inference_geo": `"us"`},
			disable: func(f *ProviderFeatureSupport) { f.InferenceGeo = false }},
		{feature: FeatureDiagnostics, name: "diagnostics", extra: map[string]any{"diagnostics": map[string]any{"cache_diagnosis": true}},
			raw:     map[string]string{"diagnostics": `{"cache_diagnosis":true}`},
			disable: func(f *ProviderFeatureSupport) { f.Diagnostics = false }},
		{feature: FeatureSafeguards, name: "safeguards", extra: map[string]any{"safeguards": json.RawMessage(`{"mode":"auto"}`)},
			raw:     map[string]string{"safeguards": `{"mode":"auto"}`},
			disable: func(f *ProviderFeatureSupport) { f.Safeguards = false }},
		{feature: FeatureCacheControlScope, name: "cache_control_scope", params: `{"cache_control":{"type":"ephemeral","scope":"global"}}`,
			disable: func(f *ProviderFeatureSupport) { f.PromptCachingScope = false }},
		{feature: FeatureFallbackCreditToken, name: "fallback_credit_token", extra: map[string]any{"fallback_credit_token": "tok"},
			raw:     map[string]string{"fallback_credit_token": `"tok"`},
			disable: func(f *ProviderFeatureSupport) { f.FallbackCredit = false }},
		{feature: FeatureContextManagement, name: "context_management", params: `{"context_management":{"edits":[{"type":"compact_20260112"}]}}`,
			raw:     map[string]string{"context_management": `{"edits":[{"type":"compact_20260112"}]}`},
			disable: func(f *ProviderFeatureSupport) { f.Compaction = false }},
		{feature: FeatureDeferLoading, name: "defer_loading", params: refusalToolParams(`"defer_loading":true`),
			disable: func(f *ProviderFeatureSupport) { f.ToolSearch = false }},
		{feature: FeatureAllowedCallers, name: "allowed_callers", params: refusalToolParams(`"allowed_callers":["direct"]`),
			disable: func(f *ProviderFeatureSupport) { f.AdvancedToolUse = false }},
		{feature: FeatureInputExamples, name: "input_examples", params: refusalToolParams(`"input_examples":[{"input":{"q":"x"}}]`),
			disable: func(f *ProviderFeatureSupport) { f.InputExamples = false }},
		{feature: FeatureEagerInputStreaming, name: "eager_input_streaming", params: refusalToolParams(`"eager_input_streaming":true`),
			disable: func(f *ProviderFeatureSupport) { f.EagerInputStreaming = false }},
		{feature: FeatureServerTool, name: "server_tool", params: `{"tools":[{"type":"web_fetch_20250910","name":"web_fetch"}]}`,
			disable: func(f *ProviderFeatureSupport) { f.WebFetch = false }},
		{feature: FeatureAnthropicBeta, name: "anthropic_beta", params: `{}`,
			disable: func(f *ProviderFeatureSupport) { f.MCP = false },
			ctx: func(ctx *schemas.BifrostContext) {
				ctx.SetValue(schemas.BifrostContextKeyClientAnthropicBeta, []string{"mcp-client-2025-11-20"})
			}},
	}
}

func refusalToolParams(flag string) string {
	return `{"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}},` + flag + `}]}`
}

// withFeatures replaces provider's ProviderFeatures entry for the test.
func withFeatures(t *testing.T, provider schemas.ModelProvider, edit func(*ProviderFeatureSupport)) {
	t.Helper()
	prev, had := ProviderFeatures[provider]
	next := prev
	edit(&next)
	ProviderFeatures[provider] = next
	t.Cleanup(func() {
		if had {
			ProviderFeatures[provider] = prev
		} else {
			delete(ProviderFeatures, provider)
		}
	})
}

func refusalCtx(mode *bool) *schemas.BifrostContext {
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	if mode != nil {
		ctx.SetValue(schemas.BifrostContextKeyRequestBodyRewriter, &schemas.RequestBodyRewriteHook{RefuseFeatures: *mode})
	}
	return ctx
}

func refusalChatRequest(t *testing.T, provider schemas.ModelProvider, params string, extra map[string]any) *schemas.BifrostChatRequest {
	t.Helper()
	p := &schemas.ChatParameters{}
	if params != "" {
		require.NoError(t, json.Unmarshal([]byte(params), p))
	}
	p.MaxCompletionTokens = schemas.Ptr(1024)
	if len(extra) > 0 {
		p.ExtraParams = map[string]any{}
		for k, v := range extra {
			p.ExtraParams[k] = v
		}
	}
	return &schemas.BifrostChatRequest{
		Provider: provider,
		Model:    refusalTestModel,
		Input:    []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("hi")}}},
		Params:   p,
	}
}

func rawExtra(raw map[string]string) map[string]any {
	out := make(map[string]any, len(raw))
	for k, v := range raw {
		out[k] = json.RawMessage(v)
	}
	return out
}

func requireRefusal(t *testing.T, bifrostErr *schemas.BifrostError, want string) {
	t.Helper()
	require.NotNil(t, bifrostErr, "want a refusal for %s", want)
	require.NotNil(t, bifrostErr.Error)
	var refusal *schemas.UnsupportedOnTargetError
	require.True(t, errors.As(bifrostErr.Error.Error, &refusal), "ErrorField.Error carries the typed refusal: %v", bifrostErr.Error.Error)
	assert.Equal(t, want, refusal.Feature)
	require.NotNil(t, bifrostErr.StatusCode)
	assert.Equal(t, 400, *bifrostErr.StatusCode)
	require.NotNil(t, bifrostErr.Error.Code)
	assert.Equal(t, schemas.ErrorCodeUnsupportedOnTarget, *bifrostErr.Error.Code)
	assert.Equal(t, want, bifrostErr.Error.Param)
	require.NotNil(t, bifrostErr.Error.Type)
	assert.Equal(t, "invalid_request_error", *bifrostErr.Error.Type)
}

// AC-8: each § Refusals row, on each Claude provider whose features lack it, is refused by the
// builder in refusal mode, typed and as a merged raw key; with refusal mode off the same request
// builds and records exactly the feature it removed.
func TestBuildChat_RefusalMode_EveryRefusalRow(t *testing.T) {
	on, off := true, false
	for _, provider := range refusalTestProviders {
		for _, row := range refusalRows() {
			variants := map[string]*schemas.BifrostChatRequest{"typed": nil}
			if len(row.raw) > 0 {
				variants["raw"] = nil
			}
			for variant := range variants {
				t.Run(string(provider)+"/"+row.name+"/"+variant, func(t *testing.T) {
					withFeatures(t, provider, row.disable)
					build := func(mode *bool) (*schemas.BifrostContext, []byte, *schemas.BifrostError) {
						ctx := refusalCtx(mode)
						if row.ctx != nil {
							row.ctx(ctx)
						}
						req := refusalChatRequest(t, provider, row.params, row.extra)
						if variant == "raw" {
							req = refusalChatRequest(t, provider, "", rawExtra(row.raw))
							ctx.SetValue(schemas.BifrostContextKeyPassthroughExtraParams, true)
						}
						body, bifrostErr := BuildAnthropicChatRequestBody(ctx, req, AnthropicRequestBuildConfig{Provider: provider, Model: refusalTestModel})
						return ctx, body, bifrostErr
					}

					_, _, bifrostErr := build(&on)
					requireRefusal(t, bifrostErr, row.name)

					ctx, body, bifrostErr := build(&off)
					require.Nil(t, bifrostErr, "refusal mode off builds")
					require.NotEmpty(t, body)
					if variant == "typed" {
						want := row.feature
						if want == FeatureAnthropicBeta {
							want = 0 // client betas are checked in refusal mode only
						}
						assert.Equal(t, want, RemovedFeatures(ctx), "the strips record exactly the removed feature")
					}
				})
			}
		}
	}
}

// The Responses builder shares the refusal point: a merged raw key, a key the converter drops at
// conversion (speed on a model without fast mode, context_management where the provider takes
// none), and a server tool the builder's validator drops.
func TestBuildResponses_RefusalMode(t *testing.T) {
	on := true
	responsesReq := func(provider schemas.ModelProvider, extra map[string]any, tools []schemas.ResponsesTool) *schemas.BifrostResponsesRequest {
		return &schemas.BifrostResponsesRequest{
			Provider: provider,
			Model:    refusalTestModel,
			Input:    []schemas.ResponsesMessage{{Role: schemas.Ptr(schemas.ResponsesInputMessageRoleUser), Content: &schemas.ResponsesMessageContent{ContentStr: schemas.Ptr("hi")}}},
			Params:   &schemas.ResponsesParameters{MaxOutputTokens: schemas.Ptr(1024), ExtraParams: extra, Tools: tools},
		}
	}
	for _, provider := range refusalTestProviders {
		for _, row := range refusalRows() {
			if len(row.raw) == 0 {
				continue
			}
			t.Run(string(provider)+"/raw/"+row.name, func(t *testing.T) {
				withFeatures(t, provider, row.disable)
				ctx := refusalCtx(&on)
				ctx.SetValue(schemas.BifrostContextKeyPassthroughExtraParams, true)
				_, bifrostErr := BuildAnthropicResponsesRequestBody(ctx, responsesReq(provider, rawExtra(row.raw), nil), AnthropicRequestBuildConfig{Provider: provider, Model: refusalTestModel})
				requireRefusal(t, bifrostErr, row.name)
			})
		}
		t.Run(string(provider)+"/conversion/speed", func(t *testing.T) {
			ctx := refusalCtx(&on)
			_, bifrostErr := BuildAnthropicResponsesRequestBody(ctx, responsesReq(provider, map[string]any{"speed": "fast"}, nil), AnthropicRequestBuildConfig{Provider: provider, Model: refusalTestModel})
			requireRefusal(t, bifrostErr, "speed")
		})
		t.Run(string(provider)+"/conversion/context_management", func(t *testing.T) {
			withFeatures(t, provider, func(f *ProviderFeatureSupport) { f.ContextManagementField = false })
			ctx := refusalCtx(&on)
			req := responsesReq(provider, nil, nil)
			req.Params.ContextManagement = json.RawMessage(`{"edits":[{"type":"compact_20260112"}]}`)
			_, bifrostErr := BuildAnthropicResponsesRequestBody(ctx, req, AnthropicRequestBuildConfig{Provider: provider, Model: refusalTestModel})
			requireRefusal(t, bifrostErr, "context_management")
		})
		t.Run(string(provider)+"/conversion/context_management_null", func(t *testing.T) {
			withFeatures(t, provider, func(f *ProviderFeatureSupport) { f.ContextManagementField = false })
			ctx := refusalCtx(&on)
			req := responsesReq(provider, map[string]any{"context_management": json.RawMessage(`null`)}, nil)
			req.Params.ContextManagement = json.RawMessage(`null`)
			_, bifrostErr := BuildAnthropicResponsesRequestBody(ctx, req, AnthropicRequestBuildConfig{Provider: provider, Model: refusalTestModel})
			require.Nil(t, bifrostErr, "a null context_management carries nothing to refuse")
		})
		t.Run(string(provider)+"/validator/server_tool", func(t *testing.T) {
			withFeatures(t, provider, func(f *ProviderFeatureSupport) { f.WebFetch = false })
			ctx := refusalCtx(&on)
			tools := []schemas.ResponsesTool{{Type: schemas.ResponsesToolTypeWebFetch}}
			_, bifrostErr := BuildAnthropicResponsesRequestBody(ctx, responsesReq(provider, nil, tools), AnthropicRequestBuildConfig{Provider: provider, Model: refusalTestModel, ValidateTools: true})
			requireRefusal(t, bifrostErr, "server_tool")
		})
	}
}

// tool_version: a Vertex tool version remap on the builder's raw path is refused; anthropic and
// azure remap no version, so the row is Vertex's only.
func TestBuildChat_RefusalMode_ToolVersionRemapOnVertex(t *testing.T) {
	on, off := true, false
	raw := []byte(`{"model":"` + refusalTestModel + `","max_tokens":64,"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"web_search_20260209","name":"web_search"}]}`)
	build := func(mode *bool) (*schemas.BifrostContext, *schemas.BifrostError) {
		ctx := refusalCtx(mode)
		ctx.SetValue(schemas.BifrostContextKeyUseRawRequestBody, true)
		req := &schemas.BifrostChatRequest{Provider: schemas.Vertex, Model: refusalTestModel, RawRequestBody: raw}
		_, bifrostErr := BuildAnthropicChatRequestBody(ctx, req, AnthropicRequestBuildConfig{Provider: schemas.Vertex, Model: refusalTestModel})
		return ctx, bifrostErr
	}
	_, bifrostErr := build(&on)
	requireRefusal(t, bifrostErr, "tool_version")
	ctx, bifrostErr := build(&off)
	require.Nil(t, bifrostErr)
	assert.Equal(t, FeatureToolVersion, RemovedFeatures(ctx))
}

// RefuseRemovedFeatures is the Vertex provider's post-build check: a tool_version its own remap
// records is refused there.
func TestRefuseRemovedFeatures_VertexRemapSite(t *testing.T) {
	on, off := true, false
	body := []byte(`{"tools":[{"type":"web_search_20260209","name":"web_search"}]}`)
	for _, mode := range []*bool{&on, &off, nil} {
		ctx := refusalCtx(mode)
		_, err := RemapRawToolVersionsForProviderReport(ctx, body, schemas.Vertex, refusalTestModel)
		require.NoError(t, err)
		refusal := RefuseRemovedFeatures(ctx, schemas.Vertex)
		if mode != nil && *mode {
			require.NotNil(t, refusal)
			assert.Equal(t, "tool_version", refusal.Feature)
			assert.Equal(t, schemas.Vertex, refusal.Provider)
		} else {
			assert.Nil(t, refusal)
		}
	}
}

// Betas from the provider's own configuration keep today's silent filter in refusal mode; only
// client-sourced values are refused, and one the provider's overrides allow counts as carried.
func TestBuildChat_RefusalMode_ConfiguredBetasFilteredSilently(t *testing.T) {
	on := true
	for _, provider := range refusalTestProviders {
		t.Run(string(provider), func(t *testing.T) {
			withFeatures(t, provider, func(f *ProviderFeatureSupport) { f.MCP = false })
			cfg := AnthropicRequestBuildConfig{Provider: provider, Model: refusalTestModel,
				ProviderExtraHeaders: map[string]schemas.SecretVar{AnthropicBetaHeader: *schemas.NewSecretVar("mcp-client-2025-11-20,unknown-beta-2099")}}
			ctx := refusalCtx(&on)
			ctx.SetValue(schemas.BifrostContextKeyExtraHeaders, map[string][]string{AnthropicBetaHeader: {"mcp-client-2025-11-20"}})
			_, bifrostErr := BuildAnthropicChatRequestBody(ctx, refusalChatRequest(t, provider, "", nil), cfg)
			require.Nil(t, bifrostErr, "configured betas are filtered, not refused")

			ctx = refusalCtx(&on)
			ctx.SetValue(schemas.BifrostContextKeyClientAnthropicBeta, []string{"mcp-client-2025-11-20"})
			cfg.BetaHeaderOverrides = map[string]bool{AnthropicMCPClientBetaHeaderPrefix: true}
			_, bifrostErr = BuildAnthropicChatRequestBody(ctx, refusalChatRequest(t, provider, "", nil), cfg)
			require.Nil(t, bifrostErr, "a client beta the overrides allow is carried")
			assert.Contains(t, MergeBetaHeaders(ctx, nil), "mcp-client-2025-11-20", "client betas are forwarded")
		})
	}
}

// Refusal mode never refuses a request that carries no unsupported feature.
func TestBuildChat_RefusalMode_CleanRequestBuilds(t *testing.T) {
	on := true
	for _, provider := range refusalTestProviders {
		ctx := refusalCtx(&on)
		ctx.SetValue(schemas.BifrostContextKeyPassthroughExtraParams, true)
		body, bifrostErr := BuildAnthropicChatRequestBody(ctx, refusalChatRequest(t, provider, `{"temperature":0.5}`, rawExtra(map[string]string{"x_unknown": `1`})), AnthropicRequestBuildConfig{Provider: provider, Model: refusalTestModel})
		require.Nil(t, bifrostErr, string(provider))
		assert.Contains(t, string(body), `"x_unknown":1`)
		assert.Zero(t, RemovedFeatures(ctx))
	}
}

// The raw strip's reporting sibling records exactly the feature each arm removes and returns the
// same bytes as the exported strip.
func TestStripUnsupportedFieldsFromRawBodyReport_RecordsExactly(t *testing.T) {
	const msgs = `"messages":[{"role":"user","content":"hi"}]`
	rows := []struct {
		body    string
		disable func(*ProviderFeatureSupport)
		want    Feature
	}{
		{`{"container":"c1",` + msgs + `}`, func(f *ProviderFeatureSupport) { f.ContainerBasic = false }, FeatureContainer},
		{`{"container":{"id":"c1","skills":[{"type":"anthropic","skill_id":"pdf"}]},` + msgs + `}`, func(f *ProviderFeatureSupport) { f.Skills = false }, FeatureContainerSkills},
		// The empty-skills clean-up removes nothing the client asked for.
		{`{"container":{"id":"c1","skills":[]},` + msgs + `}`, func(f *ProviderFeatureSupport) { f.Skills = false }, 0},
		{`{"mcp_servers":[{"type":"url","url":"u","name":"m"}],` + msgs + `}`, func(f *ProviderFeatureSupport) { f.MCP = false }, FeatureMCPServers},
		{`{"speed":"fast",` + msgs + `}`, func(f *ProviderFeatureSupport) { f.FastMode = false }, FeatureSpeed},
		{`{"output_config":{"task_budget":{"type":"tokens","total":1}},` + msgs + `}`, func(f *ProviderFeatureSupport) { f.TaskBudgets = false }, FeatureTaskBudget},
		{`{"inference_geo":"us",` + msgs + `}`, func(f *ProviderFeatureSupport) { f.InferenceGeo = false }, FeatureInferenceGeo},
		{`{"diagnostics":{"cache_diagnosis":true},` + msgs + `}`, func(f *ProviderFeatureSupport) { f.Diagnostics = false }, FeatureDiagnostics},
		{`{"safeguards":{"mode":"auto"},` + msgs + `}`, func(f *ProviderFeatureSupport) { f.Safeguards = false }, FeatureSafeguards},
		{`{"cache_control":{"type":"ephemeral","scope":"global"},` + msgs + `}`, func(f *ProviderFeatureSupport) { f.PromptCachingScope = false }, FeatureCacheControlScope},
		{`{"messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral","scope":"global"}}]}]}`, func(f *ProviderFeatureSupport) { f.PromptCachingScope = false }, FeatureCacheControlScope},
		{`{"fallback_credit_token":"t",` + msgs + `}`, func(f *ProviderFeatureSupport) { f.FallbackCredit = false }, FeatureFallbackCreditToken},
		{`{"context_management":{"edits":[{"type":"clear_tool_uses_20250919"}]},` + msgs + `}`, func(f *ProviderFeatureSupport) { f.ContextEditing = false }, FeatureContextManagement},
		{`{"context_management":{"edits":[]},` + msgs + `}`, func(f *ProviderFeatureSupport) {}, 0},
		{`{"context_management":{"edits":[{"type":"compact_20260112"}]},` + msgs + `}`, func(f *ProviderFeatureSupport) { f.ContextManagementField = false }, FeatureContextManagement},
		{`{"tools":[{"name":"f","input_schema":{},"defer_loading":true}],` + msgs + `}`, func(f *ProviderFeatureSupport) { f.ToolSearch = false }, FeatureDeferLoading},
		{`{"tools":[{"name":"f","input_schema":{},"allowed_callers":["direct"]}],` + msgs + `}`, func(f *ProviderFeatureSupport) { f.AdvancedToolUse = false }, FeatureAllowedCallers},
		{`{"tools":[{"name":"f","input_schema":{},"input_examples":[{"input":{}}]}],` + msgs + `}`, func(f *ProviderFeatureSupport) { f.InputExamples = false }, FeatureInputExamples},
		{`{"tools":[{"name":"f","input_schema":{},"eager_input_streaming":true}],` + msgs + `}`, func(f *ProviderFeatureSupport) { f.EagerInputStreaming = false }, FeatureEagerInputStreaming},
	}
	for _, row := range rows {
		t.Run(row.body, func(t *testing.T) {
			// Anthropic carries every feature, so each row turns off exactly the gate it tests.
			withFeatures(t, schemas.Anthropic, row.disable)
			ctx := refusalCtx(nil)
			got, err := StripUnsupportedFieldsFromRawBodyReport(ctx, []byte(row.body), schemas.Anthropic, refusalTestModel)
			require.NoError(t, err)
			plain, err := StripUnsupportedFieldsFromRawBody([]byte(row.body), schemas.Anthropic, refusalTestModel)
			require.NoError(t, err)
			assert.Equal(t, string(plain), string(got))
			assert.Equal(t, row.want, RemovedFeatures(ctx))
		})
	}
}

// The typed strip records exactly the feature each arm removes, and the adaptations it makes.
func TestStripUnsupportedAnthropicFields_RecordsExactly(t *testing.T) {
	rows := []struct {
		name  string
		req   AnthropicMessageRequest
		want  Feature
		adapt schemas.Adaptation
	}{
		{"inference_geo", AnthropicMessageRequest{InferenceGeo: schemas.Ptr("us")}, FeatureInferenceGeo, 0},
		{"speed", AnthropicMessageRequest{Speed: schemas.Ptr("fast")}, FeatureSpeed, 0},
		{"mcp", AnthropicMessageRequest{MCPServers: []AnthropicMCPServerV2{{Type: "url", URL: "u", Name: "m"}}}, FeatureMCPServers, 0},
		{"diagnostics", AnthropicMessageRequest{Diagnostics: &AnthropicDiagnostics{}}, FeatureDiagnostics, 0},
		{"service_tier", AnthropicMessageRequest{ServiceTier: schemas.Ptr("auto")}, 0, schemas.AdaptationServiceTier},
		{"strict", AnthropicMessageRequest{Tools: []AnthropicTool{{Name: "f", Strict: schemas.Ptr(true)}}}, 0, schemas.AdaptationStrict},
		{"cache scope", AnthropicMessageRequest{CacheControl: &schemas.CacheControl{Type: "ephemeral", Scope: schemas.Ptr("global")}}, FeatureCacheControlScope, 0},
		{"none", AnthropicMessageRequest{}, 0, 0},
	}
	for _, row := range rows {
		ctx := refusalCtx(nil)
		req := row.req
		stripUnsupportedAnthropicFields(ctx, &req, schemas.Vertex, refusalTestModel)
		assert.Equal(t, row.want, RemovedFeatures(ctx), row.name)
		assert.Equal(t, row.adapt, schemas.AdaptationsFromContext(ctx), row.name)
	}
}

// AC-9 (fork side): the post-merge raw strip on a body with no feature allocates nothing.
func TestStripMergedBody_NoFeature_AllocBudget(t *testing.T) {
	ctx := refusalCtx(nil)
	body := []byte(`{"model":"` + refusalTestModel + `","max_tokens":1024,"messages":[{"role":"user","content":"hi"}],"temperature":0.5}`)
	allocs := testing.AllocsPerRun(100, func() {
		if _, err := stripMergedBody(ctx, body, schemas.Anthropic, refusalTestModel); err != nil {
			t.Fatal(err)
		}
	})
	assert.Zero(t, allocs)
}

// The refusal error is built by providerUtils and survives errors.As through ErrorField.Error.
func TestNewBifrostUnsupportedOnTargetError(t *testing.T) {
	bifrostErr := providerUtils.NewBifrostUnsupportedOnTargetError(&schemas.UnsupportedOnTargetError{Feature: "speed", Provider: schemas.Vertex})
	requireRefusal(t, bifrostErr, "speed")
	assert.Equal(t, "speed is not supported on provider vertex", bifrostErr.Error.Message)
}

// AC-8 / AC-11 (fork side): with no hook, a hook in strip mode, or a bare rewriter, every builder's
// body is byte-identical to the base commit's (testdata golden, generated there from
// refusalOffBodies), on bodies carrying every § Refusals feature typed, as merged raw keys and as a
// raw body, for each Claude provider.
func TestBuild_RefusalModeOff_ByteIdenticalToBase(t *testing.T) {
	data, err := os.ReadFile(refusalOffGolden)
	require.NoError(t, err)
	var golden map[string]string
	require.NoError(t, json.Unmarshal(data, &golden))
	rewrite := func([]byte) (io.Reader, int64, error) { return nil, 0, nil }
	setups := map[string]func(*schemas.BifrostContext){
		"no hook": func(*schemas.BifrostContext) {},
		"hook off": func(ctx *schemas.BifrostContext) {
			ctx.SetValue(schemas.BifrostContextKeyRequestBodyRewriter, &schemas.RequestBodyRewriteHook{Rewrite: rewrite})
		},
		"hook off, nil rewrite": func(ctx *schemas.BifrostContext) {
			ctx.SetValue(schemas.BifrostContextKeyRequestBodyRewriter, &schemas.RequestBodyRewriteHook{})
		},
		"bare rewriter": func(ctx *schemas.BifrostContext) {
			ctx.SetValue(schemas.BifrostContextKeyRequestBodyRewriter, providerUtils.RequestBodyRewriter(rewrite))
		},
	}
	for name, setup := range setups {
		t.Run(name, func(t *testing.T) {
			got := refusalOffBodies(t, setup)
			require.Len(t, got, len(golden))
			for k, want := range golden {
				assert.Equal(t, want, got[k], k)
			}
		})
	}
}

// A raw fallback_credit_token on Responses (a captured unknown key) is decoded and kept where the
// provider carries it, and refused where it does not; the base dropped it at conversion.
func TestToAnthropicResponses_RawFallbackCreditToken(t *testing.T) {
	req := &schemas.BifrostResponsesRequest{
		Provider: schemas.Anthropic, Model: refusalTestModel,
		Input:  []schemas.ResponsesMessage{{Role: schemas.Ptr(schemas.ResponsesInputMessageRoleUser), Content: &schemas.ResponsesMessageContent{ContentStr: schemas.Ptr("hi")}}},
		Params: &schemas.ResponsesParameters{ExtraParams: map[string]any{"fallback_credit_token": json.RawMessage(`"tok"`)}},
	}
	out, err := ToAnthropicResponsesRequest(refusalCtx(nil), req)
	require.NoError(t, err)
	require.NotNil(t, out.FallbackCreditToken)
	assert.Equal(t, "tok", *out.FallbackCreditToken)
	assert.NotContains(t, out.ExtraParams, "fallback_credit_token")
}

// setCapsRecord installs a capability resolver answering rec for (provider, model).
func setCapsRecord(t *testing.T, provider schemas.ModelProvider, model string, rec schemas.ModelCapabilities) {
	t.Helper()
	providerUtils.SetCapabilityResolver(func(p schemas.ModelProvider, m string) *schemas.ModelCapabilities {
		if p == provider && m == model {
			return &rec
		}
		return nil
	})
	t.Cleanup(func() { providerUtils.SetCapabilityResolver(nil) })
}

// AC-8: each § Adaptations core site sets exactly its bit, in both modes, on chat.
func TestBuildChat_AdaptationSites(t *testing.T) {
	no := false
	rows := []struct {
		name     string
		provider schemas.ModelProvider
		model    string
		params   string
		record   *schemas.ModelCapabilities
		want     schemas.Adaptation
	}{
		{"sampling: refused by the model", schemas.Anthropic, refusalTestModel, `{"temperature":0.5}`, &schemas.ModelCapabilities{SupportsSamplingParams: &no}, schemas.AdaptationSampling},
		{"sampling: temperature with top_p", schemas.Anthropic, refusalTestModel, `{"temperature":0.5,"top_p":0.9}`, nil, schemas.AdaptationSampling},
		{"tool_choice: forced choice unsupported", schemas.Anthropic, refusalTestModel, `{"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"tool_choice":"required"}`, &schemas.ModelCapabilities{SupportsForcedToolChoice: &no}, schemas.AdaptationToolChoice},
		{"tool_choice: custom", schemas.Anthropic, refusalTestModel, `{"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],"tool_choice":{"type":"custom","custom":{"name":"f"}}}`, nil, schemas.AdaptationToolChoice},
		{"tool_choice: synthetic structured-output tool not forced", schemas.Vertex, refusalTestModel, `{"response_format":{"type":"json_schema","json_schema":{"name":"o","schema":{"type":"object"}}},"reasoning_effort":"high"}`, nil, schemas.AdaptationToolChoice},
		{"max_tokens: above the ceiling", schemas.Anthropic, refusalTestModel, `{"max_completion_tokens":1000000}`, nil, schemas.AdaptationMaxTokens},
		{"reasoning: display updates downgraded", schemas.Anthropic, refusalTestModel, `{"reasoning":{"effort":"high","display":"updates"}}`, nil, schemas.AdaptationReasoning},
		{"service_tier: route has none", schemas.Vertex, refusalTestModel, `{"service_tier":"auto"}`, nil, schemas.AdaptationServiceTier},
		{"strict: route has no structured outputs", schemas.Vertex, refusalTestModel, `{"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"},"strict":true}}]}`, nil, schemas.AdaptationStrict},
		{"unmapped: seed", schemas.Anthropic, refusalTestModel, `{"seed":7}`, nil, schemas.AdaptationUnmapped},
		{"unmapped: n > 1", schemas.Anthropic, refusalTestModel, `{"n":2}`, nil, schemas.AdaptationUnmapped},
		{"top_k: adaptive-only model", schemas.Anthropic, refusalTestModel, `{"top_k":5}`, &schemas.ModelCapabilities{SupportsSamplingParams: &no}, schemas.AdaptationTopK},
		{"none", schemas.Anthropic, refusalTestModel, `{"temperature":0.5}`, nil, 0},
	}
	for _, row := range rows {
		for _, mode := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/refuse=%v", row.name, mode), func(t *testing.T) {
				if row.record != nil {
					setCapsRecord(t, row.provider, row.model, *row.record)
				}
				ctx := refusalCtx(&mode)
				p := &schemas.ChatParameters{}
				require.NoError(t, json.Unmarshal([]byte(row.params), p))
				if p.MaxCompletionTokens == nil {
					p.MaxCompletionTokens = schemas.Ptr(1024)
				}
				req := &schemas.BifrostChatRequest{Provider: row.provider, Model: row.model, Params: p,
					Input: []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("hi")}}}}
				_, bifrostErr := BuildAnthropicChatRequestBody(ctx, req, AnthropicRequestBuildConfig{Provider: row.provider, Model: row.model})
				require.Nil(t, bifrostErr)
				assert.Equal(t, row.want.String(), schemas.AdaptationsFromContext(ctx).String())
			})
		}
	}
	t.Run("max_tokens: no cap sent", func(t *testing.T) {
		ctx := refusalCtx(nil)
		req := &schemas.BifrostChatRequest{Provider: schemas.Anthropic, Model: refusalTestModel,
			Input: []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("hi")}}}}
		_, bifrostErr := BuildAnthropicChatRequestBody(ctx, req, AnthropicRequestBuildConfig{Provider: schemas.Anthropic})
		require.Nil(t, bifrostErr)
		assert.Equal(t, schemas.AdaptationMaxTokens, schemas.AdaptationsFromContext(ctx))
	})
}

// AC-8: the Responses converter's adaptation sites.
func TestBuildResponses_AdaptationSites(t *testing.T) {
	no := false
	rows := []struct {
		name   string
		params schemas.ResponsesParameters
		record *schemas.ModelCapabilities
		want   schemas.Adaptation
	}{
		{"sampling", schemas.ResponsesParameters{Temperature: schemas.Ptr(0.5)}, &schemas.ModelCapabilities{SupportsSamplingParams: &no}, schemas.AdaptationSampling},
		{"tool_choice", schemas.ResponsesParameters{Tools: []schemas.ResponsesTool{{Type: schemas.ResponsesToolTypeFunction, Name: schemas.Ptr("f")}}, ToolChoice: &schemas.ResponsesToolChoice{ResponsesToolChoiceStr: schemas.Ptr("required")}}, &schemas.ModelCapabilities{SupportsForcedToolChoice: &no}, schemas.AdaptationToolChoice},
		{"max_tokens", schemas.ResponsesParameters{MaxOutputTokens: schemas.Ptr(1000000)}, nil, schemas.AdaptationMaxTokens},
		{"unmapped", schemas.ResponsesParameters{TopLogProbs: schemas.Ptr(2)}, nil, schemas.AdaptationUnmapped},
		{"top_k", schemas.ResponsesParameters{ExtraParams: map[string]any{"top_k": 5}}, &schemas.ModelCapabilities{SupportsSamplingParams: &no}, schemas.AdaptationTopK},
	}
	for _, row := range rows {
		t.Run(row.name, func(t *testing.T) {
			if row.record != nil {
				setCapsRecord(t, schemas.Anthropic, refusalTestModel, *row.record)
			}
			params := row.params
			if params.MaxOutputTokens == nil {
				params.MaxOutputTokens = schemas.Ptr(1024)
			}
			ctx := refusalCtx(nil)
			req := &schemas.BifrostResponsesRequest{Provider: schemas.Anthropic, Model: refusalTestModel, Params: &params,
				Input: []schemas.ResponsesMessage{{Role: schemas.Ptr(schemas.ResponsesInputMessageRoleUser), Content: &schemas.ResponsesMessageContent{ContentStr: schemas.Ptr("hi")}}}}
			_, bifrostErr := BuildAnthropicResponsesRequestBody(ctx, req, AnthropicRequestBuildConfig{Provider: schemas.Anthropic})
			require.Nil(t, bifrostErr)
			assert.Equal(t, row.want, schemas.AdaptationsFromContext(ctx))
		})
	}
}

// F5 (Q1 B): in refusal mode the raw model gate runs on the merged body with the builder's own
// (provider, model), so a model that is adaptive-only only by a capability fact, on Vertex, gets
// the client's raw thinking rewritten and top_k dropped, recorded as reasoning and top_k, and is
// served rather than refused. With refusal mode off the merged body is left as today.
func TestBuildChat_RefusalMode_MergedModelGate(t *testing.T) {
	no := false
	setCapsRecord(t, schemas.Vertex, refusalTestModel, schemas.ModelCapabilities{SupportsSamplingParams: &no})
	build := func(mode bool) (*schemas.BifrostContext, string) {
		ctx := refusalCtx(&mode)
		ctx.SetValue(schemas.BifrostContextKeyPassthroughExtraParams, true)
		req := refusalChatRequest(t, schemas.Vertex, "", rawExtra(map[string]string{
			"thinking": `{"type":"enabled","budget_tokens":2048}`,
			"top_k":    `5`,
		}))
		body, bifrostErr := BuildAnthropicChatRequestBody(ctx, req, AnthropicRequestBuildConfig{Provider: schemas.Vertex, Model: refusalTestModel})
		require.Nil(t, bifrostErr)
		return ctx, string(body)
	}
	ctx, body := build(true)
	assert.Contains(t, body, `"thinking":{"type":"adaptive"}`)
	assert.NotContains(t, body, `"top_k"`)
	assert.Equal(t, "reasoning, top_k", schemas.AdaptationsFromContext(ctx).String())

	_, body = build(false)
	assert.Contains(t, body, `"budget_tokens":2048`)
	assert.Contains(t, body, `"top_k":5`)
}

// AC-9 (fork side): with no unknown key and no feature, refusal mode adds no allocation to a build;
// the post-merge pass runs only on a body a merge changed.
func TestBuildChat_RefusalMode_NoAddedAllocs(t *testing.T) {
	if raceEnabled {
		t.Skip("allocation counts of pooled buffers are not stable under the race detector")
	}
	on, off := true, false
	for _, provider := range refusalTestProviders {
		req := refusalChatRequest(t, provider, refusalToolParams(`"description":"d"`), nil)
		cfg := AnthropicRequestBuildConfig{Provider: provider, Model: refusalTestModel}
		measure := func(mode *bool) float64 {
			ctx := refusalCtx(mode)
			ctx.SetValue(schemas.BifrostContextKeyPassthroughExtraParams, true)
			return testing.AllocsPerRun(50, func() {
				if _, bifrostErr := BuildAnthropicChatRequestBody(ctx, req, cfg); bifrostErr != nil {
					t.Fatal(bifrostErr.GetErrorString())
				}
			})
		}
		assert.Equal(t, measure(&off), measure(&on), string(provider))
	}
}

// A raw key carrying nothing (null, "", [] or {}) is deleted as before but is not a removed
// feature, matching the typed strip, which never sees an empty value.
func TestStripUnsupportedFieldsFromRawBodyReport_EmptyValuesNotRecorded(t *testing.T) {
	ctx := refusalCtx(nil)
	body := `{"mcp_servers":[],"speed":null,"inference_geo":"","diagnostics":{},"container":null,"fallback_credit_token":null,` +
		`"system":[{"type":"text","text":"s","cache_control":{"type":"ephemeral","scope":""}}],` +
		`"messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral","scope":null}}]}]}`
	got, err := StripUnsupportedFieldsFromRawBodyReport(ctx, []byte(body), schemas.Vertex, refusalTestModel)
	require.NoError(t, err)
	plain, err := StripUnsupportedFieldsFromRawBody([]byte(body), schemas.Vertex, refusalTestModel)
	require.NoError(t, err)
	assert.Equal(t, string(plain), string(got))
	assert.Zero(t, RemovedFeatures(ctx))

	// A nested scope with a value is still recorded, on system and on message blocks.
	for _, nested := range []string{
		`{"system":[{"type":"text","text":"s","cache_control":{"type":"ephemeral","scope":"user"}}],"messages":[]}`,
		`{"messages":[{"role":"user","content":[{"type":"text","text":"hi","cache_control":{"type":"ephemeral","scope":"user"}}]}]}`,
	} {
		ctx := refusalCtx(nil)
		_, err := StripUnsupportedFieldsFromRawBodyReport(ctx, []byte(nested), schemas.Vertex, refusalTestModel)
		require.NoError(t, err)
		assert.Equal(t, FeatureCacheControlScope, RemovedFeatures(ctx), nested)
	}
}

package anthropic

import (
	"encoding/json"
	"errors"
	"testing"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// adaptAllMask is a non-zero refuse mask holding no feature bit: every strip reports, nothing is
// refused.
const adaptAllMask = Feature(1 << 31)

// featureRow is one explicit feature: how a chat and a Responses request carry it, a target that
// lacks it, and the member that leaves the body when it is stripped.
type featureRow struct {
	feature Feature
	// chat and responses add only this member to a minimal request; responses nil means the
	// Responses shape carries the feature only as a raw key (raw).
	chat      func(*schemas.BifrostChatRequest)
	responses func(*schemas.BifrostResponsesRequest)
	// raw is the member as a top-level ExtraParams key, merged under passthrough, when it has one.
	raw map[string]string
	// body is the member inside a raw Anthropic Messages body.
	body string
	// lacks makes (provider, model) unable to carry the feature: a host that lacks it (from the
	// ledger's refused rows), or, where every Claude host carries it, a capability fixture or a
	// provider-table edit.
	provider schemas.ModelProvider
	model    string
	lacks    func(t *testing.T)
	// absent is a substring the stripped body no longer holds ("" for a header-only member).
	absent string
	ctx    func(*schemas.BifrostContext)
	betas  []string
}

const (
	featureModel  = "claude-sonnet-4-5"
	carrierModel  = "claude-opus-4-7"
	mcpClientBeta = "mcp-client-2025-11-20"
)

func chatParams(js string) func(*schemas.BifrostChatRequest) {
	return func(r *schemas.BifrostChatRequest) {
		if err := json.Unmarshal([]byte(js), r.Params); err != nil {
			panic(err)
		}
	}
}

func chatExtra(key string, v any) func(*schemas.BifrostChatRequest) {
	return func(r *schemas.BifrostChatRequest) {
		r.Params.ExtraParams = map[string]any{key: v}
	}
}

func respExtra(key string, v any) func(*schemas.BifrostResponsesRequest) {
	return func(r *schemas.BifrostResponsesRequest) {
		r.Params.ExtraParams = map[string]any{key: v}
	}
}

func respTool(js string) func(*schemas.BifrostResponsesRequest) {
	return func(r *schemas.BifrostResponsesRequest) {
		var tool schemas.ResponsesTool
		if err := json.Unmarshal([]byte(js), &tool); err != nil {
			panic(err)
		}
		r.Params.Tools = append(r.Params.Tools, tool)
	}
}

func chatReplay(block string) func(*schemas.BifrostChatRequest) {
	return func(r *schemas.BifrostChatRequest) {
		r.Input = append(r.Input, schemas.ChatMessage{
			Role:    schemas.ChatMessageRoleAssistant,
			Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("answer")},
			ChatAssistantMessage: &schemas.ChatAssistantMessage{ProviderBlocks: []schemas.ChatProviderBlock{
				{Index: 0, Provider: schemas.ProviderBlockAnthropic, Block: []byte(block)},
			}},
		}, schemas.ChatMessage{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("next")}})
	}
}

func respReplay(block string) func(*schemas.BifrostResponsesRequest) {
	return func(r *schemas.BifrostResponsesRequest) {
		var items []schemas.ResponsesMessage
		if err := json.Unmarshal([]byte(`[{"type":"message","role":"assistant","content":[`+block+`,{"type":"output_text","text":"answer"}]},`+
			`{"type":"message","role":"user","content":"next"}]`), &items); err != nil {
			panic(err)
		}
		r.Input = append(r.Input, items...)
	}
}

func withCaps(provider schemas.ModelProvider, rec schemas.ModelCapabilities) func(t *testing.T) {
	return func(t *testing.T) { setCapsRecord(t, provider, featureModel, rec) }
}

func withProviderFeatures(provider schemas.ModelProvider, edit func(*ProviderFeatureSupport)) func(t *testing.T) {
	return func(t *testing.T) { withFeatures(t, provider, edit) }
}

func featureRows() []featureRow {
	no := false
	tool := func(flag string) string {
		return `{"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}},` + flag + `}]}`
	}
	respFn := func(flag string) string {
		return `{"type":"function","name":"f","parameters":{"type":"object"},` + flag + `}`
	}
	return []featureRow{
		{feature: FeatureContainer, chat: chatParams(`{"container":"c1"}`), responses: respExtra("container", json.RawMessage(`"c1"`)),
			raw: map[string]string{"container": `"c1"`}, body: `"container":"c1"`, absent: `"container"`,
			provider: schemas.Vertex, model: featureModel, lacks: withProviderFeatures(schemas.Vertex, func(f *ProviderFeatureSupport) { f.ContainerBasic = false })},
		{feature: FeatureContainerSkills, chat: chatParams(`{"container":{"id":"c1","skills":[{"type":"anthropic","skill_id":"pdf"}]}}`),
			responses: respExtra("container", json.RawMessage(`{"id":"c1","skills":[{"type":"anthropic","skill_id":"pdf"}]}`)),
			raw:       map[string]string{"container": `{"id":"c1","skills":[{"type":"anthropic","skill_id":"pdf"}]}`},
			body:      `"container":{"id":"c1","skills":[{"type":"anthropic","skill_id":"pdf"}]}`, absent: `"skills"`,
			provider: schemas.Vertex, model: featureModel},
		{feature: FeatureMCPServers, chat: chatParams(`{"mcp_servers":[{"type":"url","url":"https://mcp.example","name":"m"}]}`),
			raw:  map[string]string{"mcp_servers": `[{"type":"url","url":"https://mcp.example","name":"m"}]`},
			body: `"mcp_servers":[{"type":"url","url":"https://mcp.example","name":"m"}]`, absent: `"mcp_servers"`,
			provider: schemas.Vertex, model: featureModel},
		{feature: FeatureSpeed, chat: chatParams(`{"speed":"fast"}`), responses: respExtra("speed", "fast"),
			raw: map[string]string{"speed": `"fast"`}, body: `"speed":"fast"`, absent: `"speed"`,
			provider: schemas.Azure, model: "claude-opus-4-6"},
		{feature: FeatureTaskBudget, chat: chatParams(`{"task_budget":{"type":"tokens","total":1000}}`),
			responses: respExtra("task_budget", json.RawMessage(`{"type":"tokens","total":1000}`)),
			body:      `"output_config":{"task_budget":{"type":"tokens","total":1000}}`, absent: `"task_budget"`,
			provider: schemas.Vertex, model: featureModel},
		{feature: FeatureInferenceGeo, chat: chatParams(`{"inference_geo":"us"}`), responses: respExtra("inference_geo", "us"),
			raw: map[string]string{"inference_geo": `"us"`}, body: `"inference_geo":"us"`, absent: `"inference_geo"`,
			provider: schemas.Vertex, model: featureModel},
		{feature: FeatureDiagnostics, chat: chatExtra("diagnostics", map[string]any{"cache_diagnosis": true}),
			responses: respExtra("diagnostics", json.RawMessage(`{"cache_diagnosis":true}`)),
			raw:       map[string]string{"diagnostics": `{"cache_diagnosis":true}`}, body: `"diagnostics":{"cache_diagnosis":true}`, absent: `"diagnostics"`,
			provider: schemas.Vertex, model: featureModel},
		{feature: FeatureSafeguards, chat: chatExtra("safeguards", json.RawMessage(`{"mode":"auto"}`)),
			responses: respExtra("safeguards", json.RawMessage(`{"mode":"auto"}`)),
			raw:       map[string]string{"safeguards": `{"mode":"auto"}`}, body: `"safeguards":{"mode":"auto"}`, absent: `"safeguards"`,
			provider: schemas.Vertex, model: featureModel, lacks: withCaps(schemas.Vertex, schemas.ModelCapabilities{SupportsSafeguards: &no})},
		{feature: FeatureCacheControlScope, chat: chatParams(`{"cache_control":{"type":"ephemeral","scope":"global"}}`),
			responses: respTool(respFn(`"cache_control":{"type":"ephemeral","scope":"global"}`)),
			body:      `"cache_control":{"type":"ephemeral","scope":"global"}`, absent: `"scope"`,
			provider: schemas.Vertex, model: featureModel},
		{feature: FeatureFallbackCreditToken, chat: chatExtra("fallback_credit_token", "tok"), responses: respExtra("fallback_credit_token", "tok"),
			raw: map[string]string{"fallback_credit_token": `"tok"`}, body: `"fallback_credit_token":"tok"`, absent: `"fallback_credit_token"`,
			provider: schemas.Vertex, model: featureModel, lacks: withProviderFeatures(schemas.Vertex, func(f *ProviderFeatureSupport) { f.FallbackCredit = false })},
		{feature: FeatureContextManagement, chat: chatParams(`{"context_management":{"edits":[{"type":"compact_20260112"}]}}`),
			responses: respExtra("context_management", json.RawMessage(`{"edits":[{"type":"compact_20260112"}]}`)),
			raw:       map[string]string{"context_management": `{"edits":[{"type":"compact_20260112"}]}`},
			body:      `"context_management":{"edits":[{"type":"compact_20260112"}]}`, absent: `"context_management"`,
			provider: schemas.Vertex, model: featureModel, lacks: withCaps(schemas.Vertex, schemas.ModelCapabilities{SupportsCompaction: &no})},
		{feature: FeatureDeferLoading, chat: chatParams(tool(`"defer_loading":true`)), responses: respTool(respFn(`"defer_loading":true`)),
			body: `"tools":[{"name":"f","input_schema":{"type":"object"},"defer_loading":true}]`, absent: `"defer_loading"`,
			provider: schemas.Vertex, model: featureModel, lacks: withCaps(schemas.Vertex, schemas.ModelCapabilities{SupportsToolSearch: &no})},
		{feature: FeatureAllowedCallers, chat: chatParams(tool(`"allowed_callers":["direct"]`)), responses: respTool(respFn(`"allowed_callers":["direct"]`)),
			body: `"tools":[{"name":"f","input_schema":{"type":"object"},"allowed_callers":["direct"]}]`, absent: `"allowed_callers"`,
			provider: schemas.Vertex, model: featureModel},
		{feature: FeatureInputExamples, chat: chatParams(tool(`"input_examples":[{"input":{"q":"x"}}]`)), responses: respTool(respFn(`"input_examples":[{"input":{"q":"x"}}]`)),
			body: `"tools":[{"name":"f","input_schema":{"type":"object"},"input_examples":[{"input":{"q":"x"}}]}]`, absent: `"input_examples"`,
			provider: schemas.Vertex, model: featureModel},
		{feature: FeatureEagerInputStreaming, chat: chatParams(tool(`"eager_input_streaming":true`)), responses: respTool(respFn(`"eager_input_streaming":true`)),
			body: `"tools":[{"name":"f","input_schema":{"type":"object"},"eager_input_streaming":true}]`, absent: `"eager_input_streaming"`,
			provider: schemas.Vertex, model: featureModel, lacks: withCaps(schemas.Vertex, schemas.ModelCapabilities{SupportsEagerInputStreaming: &no})},
		{feature: FeatureServerTool, chat: chatParams(`{"tools":[{"type":"web_fetch_20250910","name":"web_fetch"}]}`),
			responses: respTool(`{"type":"web_fetch"}`),
			body:      `"tools":[{"type":"web_fetch_20250910","name":"web_fetch"}]`, absent: `web_fetch`,
			provider: schemas.Vertex, model: featureModel},
		{feature: FeatureToolVersion, chat: chatParams(`{"tools":[{"type":"web_search_20260209","name":"web_search"}]}`),
			responses: respTool(`{"type":"web_search","web_search_version":"web_search_20260209"}`),
			body:      `"tools":[{"type":"web_search_20260209","name":"web_search"}]`, absent: `web_search_20260209`,
			provider: schemas.Vertex, model: featureModel},
		{feature: FeatureAnthropicBeta, betas: []string{mcpClientBeta},
			provider: schemas.Vertex, model: featureModel,
			ctx: func(ctx *schemas.BifrostContext) {
				ctx.SetValue(schemas.BifrostContextKeyClientAnthropicBeta, []string{mcpClientBeta})
			}},
		{feature: FeatureFallbackBlock, chat: chatReplay(`{"type":"fallback"}`), responses: respReplay(`{"type":"fallback","from_model":"a","to_model":"b"}`),
			body:   `"messages":[{"role":"assistant","content":[{"type":"fallback"}]}]`,
			absent: `"type":"fallback"`, provider: schemas.Vertex, model: featureModel},
		{feature: FeatureCompactionBlock, chat: chatReplay(`{"type":"compaction","content":"summary"}`), responses: respReplay(`{"type":"compaction","summary":"summary"}`),
			body:     `"messages":[{"role":"assistant","content":[{"type":"compaction","content":"summary"}]}]`,
			provider: schemas.Vertex, model: featureModel, lacks: withCaps(schemas.Vertex, schemas.ModelCapabilities{SupportsCompaction: &no})},
		{feature: FeatureToolChangeBlock, chat: chatReplay(`{"type":"tool_addition","tool":{"type":"tool_reference","name":"f"}}`),
			body:     `"messages":[{"role":"assistant","content":[{"type":"tool_addition","tool":{"type":"tool_reference","name":"f"}}]}]`,
			absent:   `tool_addition`,
			provider: schemas.Vertex, model: featureModel, lacks: withProviderFeatures(schemas.Vertex, func(f *ProviderFeatureSupport) { f.MidConvToolChanges = false })},
	}
}

// vertexChatPostBuild is VertexProvider.ChatCompletion's pass over a built Claude chat body
// (HostPostBuild), with a refusal answered the way the provider answers it.
func vertexChatPostBuild(t *testing.T, ctx *schemas.BifrostContext, body []byte, model string) ([]byte, *schemas.BifrostError) {
	t.Helper()
	body, err := HostPostBuild(ctx, body, schemas.Vertex, model)
	var refusal *schemas.UnsupportedOnTargetError
	if errors.As(err, &refusal) {
		return nil, providerUtils.NewBifrostUnsupportedOnTargetError(refusal)
	}
	require.NoError(t, err)
	return body, nil
}

// mergedPassKey is an unknown top-level key: merged under passthrough, it makes the typed build
// run the post-merge raw strip, the only pass that sees a replayed tool-change block.
const mergedPassKey = "x_unknown"

func featureChatRequest(row featureRow, provider schemas.ModelProvider, model string, raw bool) (*schemas.BifrostChatRequest, bool) {
	req := &schemas.BifrostChatRequest{
		Provider: provider, Model: model,
		Input:  []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("hi")}}},
		Params: &schemas.ChatParameters{MaxCompletionTokens: schemas.Ptr(1024)},
	}
	passthrough := raw || row.feature == FeatureToolChangeBlock
	switch {
	case raw:
		req.Params.ExtraParams = rawExtra(row.raw)
	case row.chat != nil:
		row.chat(req)
	}
	if row.feature == FeatureToolChangeBlock {
		if req.Params.ExtraParams == nil {
			req.Params.ExtraParams = map[string]any{}
		}
		req.Params.ExtraParams[mergedPassKey] = json.RawMessage(`1`)
	}
	return req, passthrough
}

func featureResponsesRequest(row featureRow, provider schemas.ModelProvider, model string, raw bool) *schemas.BifrostResponsesRequest {
	req := &schemas.BifrostResponsesRequest{
		Provider: provider, Model: model,
		Input:  []schemas.ResponsesMessage{{Role: schemas.Ptr(schemas.ResponsesInputMessageRoleUser), Content: &schemas.ResponsesMessageContent{ContentStr: schemas.Ptr("hi")}}},
		Params: &schemas.ResponsesParameters{MaxOutputTokens: schemas.Ptr(1024)},
	}
	switch {
	case raw:
		req.Params.ExtraParams = rawExtra(row.raw)
	case row.responses != nil:
		row.responses(req)
	}
	return req
}

func featureCtx(row featureRow, mask *Feature, passthrough bool) *schemas.BifrostContext {
	ctx := maskCtx(mask)
	if passthrough {
		ctx.SetValue(schemas.BifrostContextKeyPassthroughExtraParams, true)
	}
	if row.ctx != nil {
		row.ctx(ctx)
	}
	return ctx
}

// buildVariant is one way a row reaches a builder.
type buildVariant struct {
	name  string
	build func(t *testing.T, mask *Feature, provider schemas.ModelProvider, model string) (*schemas.BifrostContext, string, *schemas.BifrostError, FeatureAsk)
}

func rowVariants(row featureRow) []buildVariant {
	var out []buildVariant
	chat := func(raw bool) buildVariant {
		name := "chat/typed"
		if raw {
			name = "chat/raw"
		}
		return buildVariant{name, func(t *testing.T, mask *Feature, provider schemas.ModelProvider, model string) (*schemas.BifrostContext, string, *schemas.BifrostError, FeatureAsk) {
			req, passthrough := featureChatRequest(row, provider, model, raw)
			ask := ChatAsk(req, row.betas) // before the build: the converter edits the request's ExtraParams
			ctx := featureCtx(row, mask, passthrough)
			body, bifrostErr := BuildAnthropicChatRequestBody(ctx, req, AnthropicRequestBuildConfig{Provider: provider, Model: model})
			if bifrostErr == nil && provider == schemas.Vertex {
				body, bifrostErr = vertexChatPostBuild(t, ctx, body, model)
			}
			return ctx, string(body), bifrostErr, ask
		}}
	}
	responses := func(raw bool) buildVariant {
		name := "responses/typed"
		if raw {
			name = "responses/raw"
		}
		return buildVariant{name, func(t *testing.T, mask *Feature, provider schemas.ModelProvider, model string) (*schemas.BifrostContext, string, *schemas.BifrostError, FeatureAsk) {
			req := featureResponsesRequest(row, provider, model, raw)
			ask := ResponsesAsk(req, row.betas)
			ctx := featureCtx(row, mask, raw)
			// Every Claude host but Anthropic validates Responses tools (vertex, azure, bedrock).
			body, bifrostErr := BuildAnthropicResponsesRequestBody(ctx, req, AnthropicRequestBuildConfig{Provider: provider, Model: model, ValidateTools: provider != schemas.Anthropic})
			return ctx, string(body), bifrostErr, ask
		}}
	}
	if row.chat != nil || row.betas != nil {
		out = append(out, chat(false))
	}
	if len(row.raw) > 0 {
		out = append(out, chat(true), responses(true))
	}
	if row.responses != nil || row.betas != nil {
		out = append(out, responses(false))
	}
	return out
}

// AC-1: under a mask holding the feature the builder refuses it before any request is sent;
// under a mask holding every other bit it strips the member and stamps exactly that bit
// (compaction_block, reported but forwarded, keeps the member); under the zero mask the body is
// the one built with no hook at all, which TestBuild_RefusalModeOff_ByteIdenticalToBase pins to
// the upstream bytes.
func TestRefuseMask_PerFeature(t *testing.T) {
	for _, row := range featureRows() {
		for _, v := range rowVariants(row) {
			t.Run(row.feature.First()+"/"+v.name, func(t *testing.T) {
				if row.lacks != nil {
					row.lacks(t)
				}
				only := row.feature
				_, _, bifrostErr, _ := v.build(t, &only, row.provider, row.model)
				requireRefusal(t, bifrostErr, row.feature.First())

				others := ^row.feature
				ctx, body, bifrostErr, _ := v.build(t, &others, row.provider, row.model)
				require.Nil(t, bifrostErr, "a feature outside the mask is stripped, not refused")
				assert.Equal(t, row.feature, schemas.AdaptedFeaturesFromContext(ctx))
				if row.absent != "" {
					assert.NotContains(t, body, row.absent)
				}
				if row.feature == FeatureCompactionBlock {
					assert.Contains(t, body, `"compaction"`, "compaction_block is reported and forwarded")
				}

				var zero Feature
				_, zeroBody, bifrostErr, _ := v.build(t, &zero, row.provider, row.model)
				require.Nil(t, bifrostErr)
				_, noHookBody, bifrostErr, _ := v.build(t, nil, row.provider, row.model)
				require.Nil(t, bifrostErr)
				assert.Equal(t, noHookBody, zeroBody, "the zero mask is upstream behaviour")
			})
		}
	}
}

// AC-5: for every AC-1 row, and for a carrying target, the carry predicate on the detector's ask
// equals what the builder records, on every path the row reaches.
func TestUncarried_AgreesWithBuilder(t *testing.T) {
	type target struct {
		name     string
		provider schemas.ModelProvider
		model    string
		lacking  bool
	}
	for _, row := range featureRows() {
		targets := []target{
			{"lacking", row.provider, row.model, true},
			{"carrying", schemas.Anthropic, carrierModel, false},
		}
		for _, tg := range targets {
			for _, v := range rowVariants(row) {
				t.Run(row.feature.First()+"/"+tg.name+"/"+v.name, func(t *testing.T) {
					if tg.lacking && row.lacks != nil {
						row.lacks(t)
					}
					mask := adaptAllMask
					ctx, _, bifrostErr, ask := v.build(t, &mask, tg.provider, tg.model)
					require.Nil(t, bifrostErr)
					recorded := RemovedFeatures(ctx)
					assert.Equal(t, recorded, Uncarried(ask, tg.provider, tg.model, nil))
					if tg.lacking {
						assert.Equal(t, row.feature, recorded)
					} else {
						assert.Zero(t, recorded)
					}
				})
			}
		}
	}
}

// AC-5, context_management per edit kind: a compact edit needs compaction, a clear edit context
// editing, and an unknown edit type is carried everywhere.
func TestUncarried_ContextManagementEdits(t *testing.T) {
	yes, no := true, false
	models := []struct {
		name string
		rec  schemas.ModelCapabilities
		want map[string]Feature
	}{
		{"compact-only", schemas.ModelCapabilities{SupportsCompaction: &yes, SupportsContextEditing: &no},
			map[string]Feature{"compact_20260112": 0, "clear_tool_uses_20250919": FeatureContextManagement, "future_edit_2099": 0}},
		{"clear-only", schemas.ModelCapabilities{SupportsCompaction: &no, SupportsContextEditing: &yes},
			map[string]Feature{"compact_20260112": FeatureContextManagement, "clear_tool_uses_20250919": 0, "future_edit_2099": 0}},
	}
	for _, m := range models {
		for edit, want := range m.want {
			t.Run(m.name+"/"+edit, func(t *testing.T) {
				setCapsRecord(t, schemas.Anthropic, featureModel, m.rec)
				cm := `{"edits":[{"type":"` + edit + `"}]}`
				for _, raw := range []bool{false, true} {
					if !raw && edit == "future_edit_2099" {
						continue // the typed ContextManagement cannot marshal an unknown edit type
					}
					req := &schemas.BifrostChatRequest{
						Provider: schemas.Anthropic, Model: featureModel,
						Input:  []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("hi")}}},
						Params: &schemas.ChatParameters{MaxCompletionTokens: schemas.Ptr(1024)},
					}
					mask := adaptAllMask
					ctx := maskCtx(&mask)
					if raw {
						req.Params.ExtraParams = map[string]any{"context_management": json.RawMessage(cm)}
						ctx.SetValue(schemas.BifrostContextKeyPassthroughExtraParams, true)
					} else {
						req.Params.ContextManagement = json.RawMessage(cm)
					}
					ask := ChatAsk(req, nil)
					_, bifrostErr := BuildAnthropicChatRequestBody(ctx, req, AnthropicRequestBuildConfig{Provider: schemas.Anthropic, Model: featureModel})
					require.Nil(t, bifrostErr)
					assert.Equal(t, want, RemovedFeatures(ctx), "builder, raw=%v", raw)
					assert.Equal(t, want, Uncarried(ask, schemas.Anthropic, featureModel, nil), "predicate, raw=%v", raw)
					assert.Equal(t, want, Uncarried(MessagesAsk([]byte(`{"context_management":`+cm+`}`), nil), schemas.Anthropic, featureModel, nil), "messages predicate")
				}
			})
		}
	}
}

// AC-5: a request carrying only one member asks exactly that bit, on every detector; tool_version
// is not an asked bit (the tool asks server_tool and names its version in ServerTools).
func TestAsk_PerFeature(t *testing.T) {
	for _, row := range featureRows() {
		want := row.feature
		if want == FeatureToolVersion {
			want = FeatureServerTool
		}
		t.Run(row.feature.First(), func(t *testing.T) {
			if row.chat != nil {
				req, _ := featureChatRequest(row, schemas.Anthropic, featureModel, false)
				delete(req.Params.ExtraParams, mergedPassKey)
				assert.Equal(t, want, ChatAsk(req, row.betas).Features, "chat typed")
			}
			if len(row.raw) > 0 {
				req, _ := featureChatRequest(row, schemas.Anthropic, featureModel, true)
				assert.Equal(t, want, ChatAsk(req, nil).Features, "chat raw key")
				assert.Equal(t, want, ResponsesAsk(featureResponsesRequest(row, schemas.Anthropic, featureModel, true), nil).Features, "responses raw key")
			}
			if row.responses != nil {
				assert.Equal(t, want, ResponsesAsk(featureResponsesRequest(row, schemas.Anthropic, featureModel, false), nil).Features, "responses")
			}
			body := `{"model":"m","max_tokens":1,` + row.body + `}`
			if row.body == "" {
				body = `{"model":"m"}`
			}
			ask := MessagesAsk([]byte(body), row.betas)
			assert.Equal(t, want, ask.Features, "messages")
			if row.feature == FeatureToolVersion {
				assert.Equal(t, []string{"web_search_20260209"}, ask.ServerTools)
			}
		})
	}
}

// AC-5: an empty value (null, "", [] or {}) asks nothing, on a raw key and in a Messages body.
func TestAsk_EmptyValuesAskNothing(t *testing.T) {
	for _, key := range []string{"container", "mcp_servers", "speed", "inference_geo", "diagnostics", "safeguards", "fallback_credit_token", "context_management", "task_budget"} {
		for _, empty := range []string{`null`, `""`, `[]`, `{}`} {
			t.Run(key+"="+empty, func(t *testing.T) {
				req := &schemas.BifrostChatRequest{Params: &schemas.ChatParameters{ExtraParams: map[string]any{key: json.RawMessage(empty)}}}
				assert.Zero(t, ChatAsk(req, nil).Features)
				resp := &schemas.BifrostResponsesRequest{Params: &schemas.ResponsesParameters{ExtraParams: map[string]any{key: json.RawMessage(empty)}}}
				assert.Zero(t, ResponsesAsk(resp, nil).Features)
				path := key
				if key == "task_budget" {
					path = "output_config"
					empty = `{"task_budget":` + empty + `}`
				}
				assert.Zero(t, MessagesAsk([]byte(`{"`+path+`":`+empty+`}`), nil).Features)
			})
		}
	}
	assert.Zero(t, ChatAsk(nil, []string{"", " "}).Features, "blank betas ask nothing")
}

// AC-5: the detectors allocate nothing for a request that asks for nothing.
func TestAsk_NoFeature_AllocBudget(t *testing.T) {
	chat := &schemas.BifrostChatRequest{
		Input: []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("hi")}}},
		Params: &schemas.ChatParameters{MaxCompletionTokens: schemas.Ptr(64), ExtraParams: map[string]any{"top_k": 5},
			Tools: []schemas.ChatTool{{Type: schemas.ChatToolTypeFunction, Function: &schemas.ChatToolFunction{Name: "f"}}}},
	}
	resp := &schemas.BifrostResponsesRequest{
		Input:  []schemas.ResponsesMessage{{Role: schemas.Ptr(schemas.ResponsesInputMessageRoleUser), Content: &schemas.ResponsesMessageContent{ContentStr: schemas.Ptr("hi")}}},
		Params: &schemas.ResponsesParameters{MaxOutputTokens: schemas.Ptr(64), Tools: []schemas.ResponsesTool{{Type: schemas.ResponsesToolTypeFunction}}},
	}
	body := []byte(`{"model":"m","max_tokens":64,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"tools":[{"name":"f","input_schema":{"type":"object"}}]}`)
	allocs := testing.AllocsPerRun(100, func() {
		_ = ChatAsk(chat, nil)
		_ = ResponsesAsk(resp, nil)
		_ = MessagesAsk(body, nil)
	})
	assert.Zero(t, allocs)
}

// A refusal carries the feature's name in the typed error.
func TestRefuseRemovedFeatures_FirstMaskedBit(t *testing.T) {
	mask := FeatureInferenceGeo | FeatureAllowedCallers
	ctx := maskCtx(&mask)
	addRemoved(ctx, FeatureSpeed|FeatureAllowedCallers)
	refusal := RefuseRemovedFeatures(ctx, schemas.Vertex)
	require.NotNil(t, refusal)
	assert.Equal(t, "allowed_callers", refusal.Feature)
	var target *schemas.UnsupportedOnTargetError
	assert.True(t, errors.As(error(refusal), &target))
	assert.Equal(t, FeatureSpeed, schemas.AdaptedFeaturesFromContext(ctx))

	mask = FeatureInferenceGeo
	ctx = maskCtx(&mask)
	addRemoved(ctx, FeatureSpeed)
	assert.Nil(t, RefuseRemovedFeatures(ctx, schemas.Vertex))
	assert.Equal(t, "speed", schemas.AdaptedFeaturesFromContext(ctx).String())
}

// The Messages ingress leaves the typed union in ExtraParams; the Responses converter also takes a
// bare id string, so ResponsesAsk must read every form it accepts.
func TestResponsesAsk_ContainerForms(t *testing.T) {
	id := "c1"
	withSkills := AnthropicContainer{ContainerObject: &AnthropicContainerObject{ID: &id, Skills: []AnthropicContainerSkill{{}}}}
	cases := []struct {
		name string
		v    any
		want Feature
	}{
		{"typed pointer id", &AnthropicContainer{ContainerStr: &id}, FeatureContainer},
		{"typed value skills", withSkills, FeatureContainerSkills},
		{"typed pointer skills", &withSkills, FeatureContainerSkills},
		{"nil typed pointer", (*AnthropicContainer)(nil), 0},
		{"bare string", "c1", FeatureContainer},
		{"empty string", "", 0},
		{"map skills", map[string]any{"id": "c1", "skills": []any{map[string]any{"type": "anthropic"}}}, FeatureContainerSkills},
		{"map id, empty skills", map[string]any{"id": "c1", "skills": []any{}}, FeatureContainer},
		{"empty map", map[string]any{}, 0},
		{"raw skills", json.RawMessage(`{"id":"c1","skills":[{"type":"anthropic"}]}`), FeatureContainerSkills},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := &schemas.BifrostResponsesRequest{Params: &schemas.ResponsesParameters{}}
			respExtra("container", tc.v)(req)
			got := ResponsesAsk(req, nil).Features & (FeatureContainer | FeatureContainerSkills)
			if got != tc.want {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
		})
	}
}

// A decoded ExtraParams object is read in place and asks what its raw JSON form asks.
func TestAsk_DecodedExtraMatchesRaw(t *testing.T) {
	cases := []struct {
		key     string
		decoded any
		raw     string
		want    Feature
		edits   ContextEditKinds
	}{
		{"output_config", map[string]any{"task_budget": map[string]any{"type": "tokens", "total": 1000}}, `{"task_budget":{"type":"tokens","total":1000}}`, FeatureTaskBudget, 0},
		{"output_config", map[string]any{"task_budget": map[string]any{}}, `{"task_budget":{}}`, 0, 0},
		{"cache_control", map[string]any{"type": "ephemeral", "scope": "global"}, `{"type":"ephemeral","scope":"global"}`, FeatureCacheControlScope, 0},
		{"cache_control", map[string]any{"type": "ephemeral", "scope": ""}, `{"type":"ephemeral","scope":""}`, 0, 0},
		{"context_management", map[string]any{"edits": []any{map[string]any{"type": "compact_20260112"}}}, `{"edits":[{"type":"compact_20260112"}]}`, FeatureContextManagement, ContextEditCompact},
		{"context_management", map[string]any{}, `{}`, 0, 0},
	}
	mask := FeatureTaskBudget | FeatureCacheControlScope | FeatureContextManagement
	for _, tc := range cases {
		t.Run(tc.key+" "+tc.raw, func(t *testing.T) {
			for _, v := range []any{tc.decoded, json.RawMessage(tc.raw)} {
				req := &schemas.BifrostResponsesRequest{Params: &schemas.ResponsesParameters{}}
				respExtra(tc.key, v)(req)
				ask := ResponsesAsk(req, nil)
				assert.Equal(t, tc.want, ask.Features&mask, "%T", v)
				assert.Equal(t, tc.edits, ask.ContextEdits, "%T", v)
			}
		})
	}
}

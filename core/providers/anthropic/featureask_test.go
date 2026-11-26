package anthropic

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/maximhq/bifrost/core/providers/anthropic/anthropictest"
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
	featureModel = "claude-sonnet-4-5"
	carrierModel = "claude-opus-4-7"
)

func withCaps(provider schemas.ModelProvider, rec schemas.ModelCapabilities) func(t *testing.T) {
	return func(t *testing.T) { setCapsRecord(t, provider, featureModel, rec) }
}

func withProviderFeatures(provider schemas.ModelProvider, edit func(*ProviderFeatureSupport)) func(t *testing.T) {
	return func(t *testing.T) { withFeatures(t, provider, edit) }
}

// featureRows are anthropictest.FeatureRows with each feature's Messages body, stripped-body
// probe and lacking target.
func featureRows() []featureRow {
	no := false
	vertex := func(row featureRow) featureRow {
		row.provider, row.model = schemas.Vertex, featureModel
		return row
	}
	local := map[string]featureRow{
		"container": vertex(featureRow{body: `"container":"cprobe1"`, absent: `"container"`,
			lacks: withProviderFeatures(schemas.Vertex, func(f *ProviderFeatureSupport) { f.ContainerBasic = false })}),
		"container_skills": vertex(featureRow{body: `"container":{"id":"c1","skills":[{"type":"anthropic","skill_id":"skillprobe"}]}`, absent: `"skills"`}),
		"mcp_servers":      vertex(featureRow{body: `"mcp_servers":[{"type":"url","url":"https://mcpprobe.example","name":"m"}]`, absent: `"mcp_servers"`}),
		"speed":            {body: `"speed":"fast"`, absent: `"speed"`, provider: schemas.Azure, model: "claude-opus-4-6"},
		"task_budget":      vertex(featureRow{body: `"output_config":{"task_budget":{"type":"tokens","total":4242}}`, absent: `"task_budget"`}),
		"inference_geo":    vertex(featureRow{body: `"inference_geo":"us"`, absent: `"inference_geo"`}),
		"diagnostics":      vertex(featureRow{body: `"diagnostics":{"cache_diagnosis":true}`, absent: `"diagnostics"`}),
		"safeguards": vertex(featureRow{body: `"safeguards":{"mode":"auto"}`, absent: `"safeguards"`,
			lacks: withCaps(schemas.Vertex, schemas.ModelCapabilities{SupportsSafeguards: &no})}),
		"cache_control_scope": vertex(featureRow{body: `"cache_control":{"type":"ephemeral","scope":"global"}`, absent: `"scope"`}),
		"fallback_credit_token": vertex(featureRow{body: `"fallback_credit_token":"tokprobe"`, absent: `"fallback_credit_token"`,
			lacks: withProviderFeatures(schemas.Vertex, func(f *ProviderFeatureSupport) { f.FallbackCredit = false })}),
		"context_management": vertex(featureRow{body: `"context_management":{"edits":[{"type":"compact_20260112"}]}`, absent: `"context_management"`,
			lacks: withCaps(schemas.Vertex, schemas.ModelCapabilities{SupportsCompaction: &no})}),
		"defer_loading": vertex(featureRow{body: `"tools":[{"name":"f","input_schema":{"type":"object"},"defer_loading":true}]`, absent: `"defer_loading"`,
			lacks: withCaps(schemas.Vertex, schemas.ModelCapabilities{SupportsToolSearch: &no})}),
		"allowed_callers": vertex(featureRow{body: `"tools":[{"name":"f","input_schema":{"type":"object"},"allowed_callers":["direct"]}]`, absent: `"allowed_callers"`}),
		"input_examples":  vertex(featureRow{body: `"tools":[{"name":"f","input_schema":{"type":"object"},"input_examples":[{"input":{"q":"x"}}]}]`, absent: `"input_examples"`}),
		"eager_input_streaming": vertex(featureRow{body: `"tools":[{"name":"f","input_schema":{"type":"object"},"eager_input_streaming":true}]`, absent: `"eager_input_streaming"`,
			lacks: withCaps(schemas.Vertex, schemas.ModelCapabilities{SupportsEagerInputStreaming: &no})}),
		"server_tool":  vertex(featureRow{body: `"tools":[{"type":"web_fetch_20250910","name":"web_fetch"}]`, absent: `web_fetch`}),
		"tool_version": vertex(featureRow{body: `"tools":[{"type":"web_search_20260209","name":"web_search"}]`, absent: `web_search_20260209`}),
		"anthropic_beta": vertex(featureRow{ctx: func(ctx *schemas.BifrostContext) {
			ctx.SetValue(schemas.BifrostContextKeyClientAnthropicBeta, []string{anthropictest.MCPClientBeta})
		}}),
		"fallback_block": vertex(featureRow{body: `"messages":[{"role":"assistant","content":[{"type":"fallback"}]}]`, absent: `"type":"fallback"`}),
		"compaction_block": vertex(featureRow{body: `"messages":[{"role":"assistant","content":[{"type":"compaction","content":"summaryprobe"}]}]`,
			lacks: withCaps(schemas.Vertex, schemas.ModelCapabilities{SupportsCompaction: &no})}),
		"tool_change_block": vertex(featureRow{body: `"messages":[{"role":"assistant","content":[{"type":"tool_addition","tool":{"type":"tool_reference","name":"f"}}]}]`,
			absent: `tool_addition`,
			lacks:  withProviderFeatures(schemas.Vertex, func(f *ProviderFeatureSupport) { f.MidConvToolChanges = false })}),
	}
	shared := anthropictest.FeatureRows()
	rows := make([]featureRow, 0, len(shared))
	for _, s := range shared {
		row, ok := local[s.Name]
		if !ok {
			panic("featureRows: no local row for " + s.Name)
		}
		row.feature, row.chat, row.responses, row.raw, row.betas = s.Feature(), s.Chat, s.Responses, s.Raw, s.Betas
		rows = append(rows, row)
	}
	return rows
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
			anthropictest.RespExtra("container", tc.v)(req)
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
				anthropictest.RespExtra(tc.key, v)(req)
				ask := ResponsesAsk(req, nil)
				assert.Equal(t, tc.want, ask.Features&mask, "%T", v)
				assert.Equal(t, tc.edits, ask.ContextEdits, "%T", v)
			}
		})
	}
}

// responsesPromises are the explicit features a raw Responses forward must refuse.
const responsesPromises = FeatureInferenceGeo | FeatureAllowedCallers | FeatureCompactionBlock

// responsesWireBody is the client body a Responses request decodes from: its parameters, its
// input and its ExtraParams as top-level keys.
func responsesWireBody(t *testing.T, req *schemas.BifrostResponsesRequest) []byte {
	t.Helper()
	params, err := json.Marshal(req.Params)
	require.NoError(t, err)
	top := map[string]any{}
	require.NoError(t, json.Unmarshal(params, &top))
	for k, v := range req.Params.ExtraParams {
		top[k] = v
	}
	top["model"], top["input"] = "m", req.Input
	body, err := json.Marshal(top)
	require.NoError(t, err)
	return body
}

// The raw-body detector agrees with ResponsesAsk on every feature row: the same features, context
// edits and server tools from the body the request decodes from.
func TestResponsesBodyAsk_AgreesWithResponsesAsk(t *testing.T) {
	for _, row := range featureRows() {
		for _, raw := range []bool{false, true} {
			if (raw && len(row.raw) == 0) || (!raw && row.responses == nil) {
				continue
			}
			t.Run(row.feature.First()+map[bool]string{false: "/typed", true: "/raw"}[raw], func(t *testing.T) {
				req := featureResponsesRequest(row, schemas.OpenAI, "gpt-4.1", raw)
				want := ResponsesAsk(req, row.betas)
				got := ResponsesBodyAsk(responsesWireBody(t, req), row.betas)
				assert.Equal(t, want.Features, got.Features)
				assert.Equal(t, want.ContextEdits, got.ContextEdits)
				assert.Equal(t, want.ServerTools, got.ServerTools)
			})
		}
	}
	for _, empty := range []string{`null`, `""`, `[]`, `{}`} {
		assert.Zero(t, ResponsesBodyAsk([]byte(`{"inference_geo":`+empty+`,"tools":[{"type":"function","name":"f","allowed_callers":`+empty+`}]}`), nil).Features, empty)
	}
	assert.Zero(t, ResponsesBodyAsk([]byte(`{"input":"compaction \"compaction\""}`), nil).Features, "a string input holds no block")
	assert.Equal(t, FeatureAnthropicBeta, ResponsesBodyAsk([]byte(`{"model":"m","input":"hi"}`), []string{"b"}).Features, "the betas ask anthropic_beta")
	for _, tools := range []string{
		`[{"type":"web_fetch_20250910","name":"web_fetch"}]`,
		`[{"type":"web_search_20260209","name":"web_search","web_search_version":"web_search_20260209"}]`,
		`[{"type":"code_execution_20250825","name":"code_execution"}]`,
	} {
		var decoded []schemas.ResponsesTool
		require.NoError(t, json.Unmarshal([]byte(tools), &decoded))
		want := ResponsesAsk(&schemas.BifrostResponsesRequest{Params: &schemas.ResponsesParameters{Tools: decoded}}, nil)
		require.NotZero(t, want.Features&FeatureServerTool, tools)
		got := ResponsesBodyAsk([]byte(`{"model":"m","input":"hi","tools":`+tools+`}`), nil)
		assert.Equal(t, want.Features, got.Features, "a dated type is classified as the decode normalizes it: %s", tools)
		assert.Equal(t, want.ServerTools, got.ServerTools, tools)
	}
}

// The raw-body detector allocates nothing for a body that asks for nothing, nor for one that asks
// only member-level features.
func TestResponsesBodyAsk_AllocBudget(t *testing.T) {
	plain := []byte(`{"model":"m","input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]}],"tools":[{"type":"function","name":"f"}]}`)
	asking := []byte(`{"model":"m","inference_geo":"us","input":[{"role":"assistant","content":[{"type":"compaction","summary":"s"}]}],"tools":[{"type":"function","name":"f","allowed_callers":["direct"]}]}`)
	allocs := testing.AllocsPerRun(100, func() {
		_ = ResponsesBodyAsk(plain, nil)
		_ = ResponsesBodyAsk(asking, nil)
	})
	assert.Zero(t, allocs)
	assert.Equal(t, responsesPromises, ResponsesBodyAsk(asking, nil).Features)
	assert.Zero(t, ResponsesBodyAsk(plain, nil).Features)
}

// ExtraKeyFeature asks what ChatAsk asks of the same single ExtraParams key, on every feature
// row's raw key, and allocates nothing.
func TestExtraKeyFeature_AgreesWithChatAsk(t *testing.T) {
	for _, row := range featureRows() {
		for k, v := range rawExtra(row.raw) {
			req := &schemas.BifrostChatRequest{Params: &schemas.ChatParameters{ExtraParams: map[string]any{k: v}}}
			assert.Equal(t, ChatAsk(req, nil).Features, ExtraKeyFeature(k, v), k)
			assert.NotZero(t, ExtraKeyFeature(k, v), k)
		}
	}
	assert.Zero(t, ExtraKeyFeature("top_k", 5))
	var geo any = json.RawMessage(`"us"`)
	allocs := testing.AllocsPerRun(100, func() {
		_ = ExtraKeyFeature("inference_geo", geo)
		_ = ExtraKeyFeature("top_k", 5)
	})
	assert.Zero(t, allocs)
}

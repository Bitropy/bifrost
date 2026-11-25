package anthropic

import (
	"bytes"
	"encoding/json"
	"slices"
	"strings"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/tidwall/gjson"
)

// featureGate answers whether one target (provider and model) carries an explicit feature. The
// strips and Uncarried ask the same gate, so the prediction cannot drift from what is stripped.
type featureGate struct {
	provider schemas.ModelProvider
	features ProviderFeatureSupport
	caps     schemas.ModelCaps
}

// carries reports whether the target carries the member-level feature f (one bit). Context
// management, server tools, tool versions and betas are gated per edit, tool or value
// (carriesEdits, Uncarried).
func (g featureGate) carries(f Feature) bool {
	switch f {
	case FeatureContainer:
		return g.features.ContainerBasic
	case FeatureContainerSkills:
		return g.caps.SupportsSkills(g.features.Skills)
	case FeatureMCPServers:
		return g.caps.SupportsMCP(g.features.MCP)
	case FeatureSpeed:
		return g.features.FastMode && g.caps.SupportsFastMode(DefaultSupportsFastMode(g.provider, g.caps.Model()))
	case FeatureTaskBudget:
		return g.caps.SupportsTaskBudgets(g.features.TaskBudgets)
	case FeatureInferenceGeo:
		return g.caps.SupportsInferenceGeo(g.features.InferenceGeo)
	case FeatureDiagnostics:
		return g.features.Diagnostics
	case FeatureSafeguards:
		return g.features.Safeguards && g.caps.SupportsSafeguards(DefaultSupportsSafeguards(g.provider, g.caps.Model()))
	case FeatureCacheControlScope:
		return g.features.PromptCachingScope
	case FeatureFallbackCreditToken:
		return g.features.FallbackCredit
	case FeatureDeferLoading:
		return g.caps.SupportsToolSearch(g.features.ToolSearch)
	case FeatureAllowedCallers:
		return g.caps.SupportsAdvancedToolUse(g.features.AdvancedToolUse)
	case FeatureInputExamples:
		return g.caps.SupportsInputExamples(g.features.InputExamples)
	case FeatureEagerInputStreaming:
		return g.caps.SupportsEagerInputStreaming(g.features.EagerInputStreaming)
	case FeatureFallbackBlock:
		return g.features.ServerSideFallback
	case FeatureCompactionBlock:
		return g.caps.SupportsCompaction(g.features.Compaction)
	case FeatureToolChangeBlock:
		return g.features.MidConvToolChanges
	}
	return true
}

// carriesEdits reports whether the target keeps every context_management edit kind in k. An
// unknown edit type is kept for the upstream to reject.
func (g featureGate) carriesEdits(k ContextEditKinds) bool {
	if k&ContextEditCompact != 0 && !g.caps.SupportsCompaction(g.features.Compaction) {
		return false
	}
	return k&ContextEditClear == 0 || g.caps.SupportsContextEditing(g.features.ContextEditing)
}

// ContextEditKinds is a set of context_management.edits[].type kinds.
type ContextEditKinds uint8

const (
	// ContextEditCompact is a compact edit, gated by compaction.
	ContextEditCompact ContextEditKinds = 1 << iota
	// ContextEditClear is a clear_tool_uses or clear_thinking edit, gated by context editing.
	ContextEditClear
	// ContextEditUnknown is any other edit type; the strips keep it.
	ContextEditUnknown
)

// contextEditKind classifies one edit type.
func contextEditKind(t string) ContextEditKinds {
	switch ContextManagementEditType(t) {
	case ContextManagementEditTypeCompact:
		return ContextEditCompact
	case ContextManagementEditTypeClearToolUses, ContextManagementEditTypeClearThinking:
		return ContextEditClear
	}
	return ContextEditUnknown
}

// FeatureAsk is what a request asks for in explicit Anthropic features, before any target is
// chosen. ServerTools are the client's server-tool type strings (a Responses tool's type, or its
// client-selected version when it has one); Betas are the client's anthropic-beta values;
// ContextEdits are the context_management edit kinds sent. tool_version is never an asked bit: it
// is a property of a server tool on a target, which Uncarried reads off ServerTools.
type FeatureAsk struct {
	Features     Feature
	ServerTools  []string
	Betas        []string
	ContextEdits ContextEditKinds
}

// ChatAsk is the FeatureAsk of a chat request: its typed members, its ExtraParams keys (the
// paths the raw strip gates), its tools and its replayed provider blocks. Allocation-free when
// the request asks for nothing.
func ChatAsk(req *schemas.BifrostChatRequest, betas []string) FeatureAsk {
	ask := askBetas(betas)
	if req == nil {
		return ask
	}
	if p := req.Params; p != nil {
		if c := p.Container; c != nil {
			skills := 0
			if c.ContainerObject != nil {
				skills = len(c.ContainerObject.Skills)
			}
			ask.addContainer(c.ContainerStr, c.ContainerObject != nil, skills)
		}
		ask.Features |= when(len(p.MCPServers) > 0, FeatureMCPServers) |
			when(p.Speed != nil && *p.Speed != "", FeatureSpeed) |
			when(p.TaskBudget != nil, FeatureTaskBudget) |
			when(p.InferenceGeo != nil && *p.InferenceGeo != "", FeatureInferenceGeo) |
			when(scoped(p.CacheControl), FeatureCacheControlScope)
		ask.addContextManagement(p.ContextManagement)
		ask.addExtra(p.ExtraParams)
		for i := range p.Tools {
			t := &p.Tools[i]
			ask.Features |= toolFlags(t.DeferLoading != nil, len(t.AllowedCallers) > 0, len(t.InputExamples) > 0, t.EagerInputStreaming != nil, scoped(t.CacheControl))
			if t.Function == nil && t.Custom == nil && t.Type != "" && t.Type != schemas.ChatToolTypeFunction && t.Type != schemas.ChatToolTypeCustom {
				ask.Features |= FeatureServerTool
				ask.ServerTools = append(ask.ServerTools, string(t.Type))
			}
		}
	}
	for i := range req.Input {
		ask.addChatMessage(&req.Input[i])
	}
	return ask
}

func (ask *FeatureAsk) addChatMessage(m *schemas.ChatMessage) {
	if m.Content != nil {
		for i := range m.Content.ContentBlocks {
			ask.Features |= when(scoped(m.Content.ContentBlocks[i].CacheControl), FeatureCacheControlScope)
		}
	}
	if m.ChatAssistantMessage == nil {
		return
	}
	for i := range m.ChatAssistantMessage.ToolCalls {
		ask.Features |= when(scoped(m.ChatAssistantMessage.ToolCalls[i].CacheControl), FeatureCacheControlScope)
	}
	for i := range m.ChatAssistantMessage.ProviderBlocks {
		ask.Features |= replayBlockFeatures(gjson.Parse(bytesAsString(m.ChatAssistantMessage.ProviderBlocks[i].Block)))
	}
}

// ResponsesAsk is ChatAsk for a Responses request: its ExtraParams keys (where the Responses
// shape carries the Anthropic members), context_management, its tools and its input items.
func ResponsesAsk(req *schemas.BifrostResponsesRequest, betas []string) FeatureAsk {
	ask := askBetas(betas)
	if req == nil {
		return ask
	}
	if p := req.Params; p != nil {
		ask.addContextManagement(p.ContextManagement)
		ask.addExtra(p.ExtraParams)
		for i := range p.Tools {
			t := &p.Tools[i]
			ask.Features |= toolFlags(t.DeferLoading != nil, len(t.AllowedCallers) > 0, len(t.InputExamples) > 0, t.EagerInputStreaming != nil, scoped(t.CacheControl))
			if isResponsesServerTool(t.Type) {
				ask.Features |= FeatureServerTool
				ask.ServerTools = append(ask.ServerTools, responsesServerToolType(t))
			}
		}
	}
	for i := range req.Input {
		ask.addResponsesItem(&req.Input[i])
	}
	return ask
}

// isResponsesServerTool reports whether a Responses tool type is one the validator gates; the
// gate's answer is not needed, so it is asked with an empty target.
func isResponsesServerTool(t schemas.ResponsesToolType) bool {
	_, known := responsesToolSupported(t, ProviderFeatureSupport{}, schemas.ModelCaps{})
	return known
}

// responsesServerToolType is the string Uncarried gates a Responses server tool by: its
// client-selected version when it has one (a dated Anthropic type, gated by the same validator
// prefix), else its Responses type.
func responsesServerToolType(t *schemas.ResponsesTool) string {
	var version *string
	switch {
	case t.Type == schemas.ResponsesToolTypeWebSearch && t.ResponsesToolWebSearch != nil:
		version = t.ResponsesToolWebSearch.Version
	case t.Type == schemas.ResponsesToolTypeWebFetch && t.ResponsesToolWebFetch != nil:
		version = t.ResponsesToolWebFetch.Version
	}
	if version != nil && *version != "" {
		return *version
	}
	return string(t.Type)
}

func (ask *FeatureAsk) addResponsesItem(m *schemas.ResponsesMessage) {
	if m.Content == nil {
		return
	}
	for i := range m.Content.ContentBlocks {
		b := &m.Content.ContentBlocks[i]
		ask.Features |= when(scoped(b.CacheControl), FeatureCacheControlScope)
		switch b.Type {
		case schemas.ResponsesOutputMessageContentTypeFallback:
			ask.Features |= FeatureFallbackBlock
		case schemas.ResponsesOutputMessageContentTypeCompaction:
			ask.Features |= FeatureCompactionBlock
		case schemas.ResponsesInputMessageContentBlockTypeToolAddition, schemas.ResponsesInputMessageContentBlockTypeToolRemoval:
			ask.Features |= FeatureToolChangeBlock
		}
	}
}

// MessagesAsk is ChatAsk for a raw Anthropic Messages body, read at the raw strip's paths without
// decoding it.
func MessagesAsk(body []byte, betas []string) FeatureAsk {
	ask := askBetas(betas)
	if len(body) == 0 {
		return ask
	}
	doc := bytesAsString(body)
	ask.addRawContainer(gjson.Get(doc, "container"))
	for _, m := range rawMemberFeatures {
		ask.Features |= when(rawMeaningful(gjson.Get(doc, m.path)), m.feature)
	}
	if cm := gjson.Get(doc, "context_management"); rawMeaningful(cm) {
		ask.Features |= FeatureContextManagement
		ask.ContextEdits |= rawContextEdits(cm)
	}
	gjson.Get(doc, "tools").ForEach(func(_, tool gjson.Result) bool {
		ask.Features |= toolFlags(rawPresent(tool.Get("defer_loading")), rawMeaningful(tool.Get("allowed_callers")),
			rawMeaningful(tool.Get("input_examples")), rawPresent(tool.Get("eager_input_streaming")), rawMeaningful(tool.Get("cache_control.scope")))
		if t := tool.Get("type").Str; t != "" && t != "custom" {
			ask.Features |= FeatureServerTool
			ask.ServerTools = append(ask.ServerTools, t)
		}
		return true
	})
	if bytes.Contains(body, cacheControlScopeKeyToken) {
		gjson.Get(doc, "system").ForEach(func(_, block gjson.Result) bool {
			ask.Features |= when(rawMeaningful(block.Get("cache_control.scope")), FeatureCacheControlScope)
			return true
		})
	}
	if bytes.Contains(body, cacheControlScopeKeyToken) || bytes.Contains(body, fallbackBlockTypeToken) ||
		bytes.Contains(body, compactionBlockTypeToken) || mayCarryToolChangeBlocks(body) {
		gjson.Get(doc, "messages").ForEach(func(_, msg gjson.Result) bool {
			msg.Get("content").ForEach(func(_, block gjson.Result) bool {
				ask.Features |= replayBlockFeatures(block) | when(rawMeaningful(block.Get("cache_control.scope")), FeatureCacheControlScope)
				return true
			})
			return true
		})
	}
	return ask
}

// rawMemberFeatures are the top-level member paths the raw strip gates one feature at, other than
// container and context_management, whose bit depends on the value.
var rawMemberFeatures = [...]struct {
	path    string
	feature Feature
}{
	{"mcp_servers", FeatureMCPServers},
	{"speed", FeatureSpeed},
	{"output_config.task_budget", FeatureTaskBudget},
	{"inference_geo", FeatureInferenceGeo},
	{"diagnostics", FeatureDiagnostics},
	{"safeguards", FeatureSafeguards},
	{"cache_control.scope", FeatureCacheControlScope},
	{"fallback_credit_token", FeatureFallbackCreditToken},
}

func askBetas(betas []string) FeatureAsk {
	for _, b := range betas {
		if strings.TrimSpace(b) != "" {
			return FeatureAsk{Features: FeatureAnthropicBeta, Betas: betas}
		}
	}
	return FeatureAsk{}
}

func when(cond bool, f Feature) Feature {
	if cond {
		return f
	}
	return 0
}

func toolFlags(deferLoading, allowedCallers, inputExamples, eager, scope bool) Feature {
	return when(deferLoading, FeatureDeferLoading) | when(allowedCallers, FeatureAllowedCallers) |
		when(inputExamples, FeatureInputExamples) | when(eager, FeatureEagerInputStreaming) |
		when(scope, FeatureCacheControlScope)
}

func scoped(cc *schemas.CacheControl) bool {
	return cc != nil && cc.Scope != nil && *cc.Scope != ""
}

// rawPresent reports whether v exists and is not null: the typed strips remove a flag that is set
// at all, false included.
func rawPresent(v gjson.Result) bool {
	return v.Exists() && v.Type != gjson.Null
}

// replayBlockFeatures is the replay-block bit of one content block.
func replayBlockFeatures(block gjson.Result) Feature {
	t := AnthropicContentBlockType(block.Get("type").Str)
	switch {
	case t == AnthropicContentBlockTypeFallback:
		return FeatureFallbackBlock
	case t == AnthropicContentBlockTypeCompaction:
		return FeatureCompactionBlock
	case isToolChangeBlock(t):
		return FeatureToolChangeBlock
	}
	return 0
}

// addContainer reads the container union in its typed forms (chat's ChatContainer and the
// AnthropicContainer the Messages ingress leaves in ExtraParams share its shape).
func (ask *FeatureAsk) addContainer(str *string, object bool, skills int) {
	switch {
	case object && skills > 0:
		ask.Features |= FeatureContainerSkills
	case object || (str != nil && *str != ""):
		ask.Features |= FeatureContainer
	}
}

// addExtraContainer reads every form the Responses converter accepts under ExtraParams
// "container": the typed union, a bare id string, or JSON.
func (ask *FeatureAsk) addExtraContainer(v any) {
	switch c := v.(type) {
	case *AnthropicContainer:
		if c != nil {
			ask.addTypedContainer(*c)
		}
	case AnthropicContainer:
		ask.addTypedContainer(c)
	case string:
		ask.addContainer(&c, false, 0)
	default:
		ask.addRawContainer(extraResult(v))
	}
}

func (ask *FeatureAsk) addTypedContainer(c AnthropicContainer) {
	skills := 0
	if c.ContainerObject != nil {
		skills = len(c.ContainerObject.Skills)
	}
	ask.addContainer(c.ContainerStr, c.ContainerObject != nil, skills)
}

func (ask *FeatureAsk) addRawContainer(c gjson.Result) {
	switch {
	case c.IsObject() && rawMeaningful(c.Get("skills")):
		ask.Features |= FeatureContainerSkills
	case rawMeaningful(c):
		ask.Features |= FeatureContainer
	}
}

func (ask *FeatureAsk) addContextManagement(v any) {
	if !meaningful(v) {
		return
	}
	ask.Features |= FeatureContextManagement
	switch v := v.(type) {
	case json.RawMessage:
		ask.ContextEdits |= rawContextEdits(gjson.Parse(bytesAsString(v)))
	case *ContextManagement:
		for _, e := range v.Edits {
			ask.ContextEdits |= contextEditKind(string(e.Type))
		}
	case string:
		ask.ContextEdits |= rawContextEdits(gjson.Parse(v))
	}
}

func rawContextEdits(cm gjson.Result) ContextEditKinds {
	var k ContextEditKinds
	cm.Get("edits").ForEach(func(_, e gjson.Result) bool {
		k |= contextEditKind(e.Get("type").Str)
		return true
	})
	return k
}

// addExtra reads the feature members an embedder left in ExtraParams: values the converters
// promote, and raw keys merged onto the body under passthrough.
func (ask *FeatureAsk) addExtra(extra map[string]any) {
	if len(extra) == 0 {
		return
	}
	if c, ok := extra["container"]; ok {
		ask.addExtraContainer(c)
	}
	for _, m := range rawMemberFeatures {
		top, rest, nested := strings.Cut(m.path, ".")
		v, ok := extra[top]
		if !ok {
			continue
		}
		if !nested {
			ask.Features |= when(meaningful(v), m.feature)
			continue
		}
		ask.Features |= when(rawMeaningful(extraResult(v).Get(rest)), m.feature)
	}
	// The Responses converter reads a top-level task_budget key.
	if v, ok := extra["task_budget"]; ok && meaningful(v) {
		ask.Features |= FeatureTaskBudget
	}
	if v, ok := extra["context_management"]; ok {
		if r := extraResult(v); r.Exists() {
			ask.addContextManagement(json.RawMessage(r.Raw))
		} else {
			ask.addContextManagement(v)
		}
	}
}

// extraResult views an ExtraParams value as JSON without copying when it already is JSON (raw
// bytes) and re-encodes a decoded object; any other value yields an empty result.
func extraResult(v any) gjson.Result {
	switch v := v.(type) {
	case json.RawMessage:
		return gjson.Parse(bytesAsString(v))
	case map[string]any:
		// A decoded object: re-encode once; only on a request that sends one of these keys.
		if b, err := json.Marshal(v); err == nil {
			return gjson.ParseBytes(b)
		}
	}
	return gjson.Result{}
}

// Uncarried is the carry predicate: the bits of ask the Anthropic-family builders would record in
// BifrostContextKeyRemovedFeatures for provider (the attempt's base provider) and model (its
// canonical model). betaOverrides is the configured provider's map. It asks the strips' own
// gates (featureGate), the tool validators and FilterBetaHeadersForProvider.
func Uncarried(ask FeatureAsk, provider schemas.ModelProvider, model string, betaOverrides map[string]bool) Feature {
	features, ok := ProviderFeatures[provider]
	if !ok {
		return 0
	}
	g := featureGate{provider: provider, features: features, caps: schemas.ResolveModelCaps(provider, model)}
	var out Feature
	for rest := ask.Features &^ (FeatureContextManagement | FeatureServerTool | FeatureToolVersion | FeatureAnthropicBeta); rest != 0; rest &= rest - 1 {
		if bit := rest & -rest; !g.carries(bit) {
			out |= bit
		}
	}
	if ask.Features&FeatureContextManagement != 0 && (!features.ContextManagementField || !g.carriesEdits(ask.ContextEdits)) {
		out |= FeatureContextManagement
	}
	for _, t := range ask.ServerTools {
		if !g.carriesServerTool(t) {
			out |= FeatureServerTool
		} else if _, remapped := remappedToolVersion(provider, g.sentToolType(t)); remapped {
			out |= FeatureToolVersion
		}
	}
	if len(ask.Betas) > 0 {
		merged := mergeBetaValues(nil, slices.Values(ask.Betas))
		if len(FilterBetaHeadersForProvider(merged, provider, betaOverrides)) < len(merged) {
			out |= FeatureAnthropicBeta
		}
	}
	return out
}

// carriesServerTool is the validators' gate for one server-tool string: a Responses tool type, or
// an Anthropic tool type.
func (g featureGate) carriesServerTool(t string) bool {
	if supported, known := responsesToolSupported(schemas.ResponsesToolType(t), g.features, g.caps); known {
		return supported
	}
	return isAnthropicServerToolSupported(t, g.features, g.caps)
}

// sentToolType is the type the chat raw path sends for an Anthropic tool type before the host
// remap: the computer-use family is normalized to the model's generation first.
func (g featureGate) sentToolType(t string) string {
	base := computerUseBaseTool(t)
	if base == "" || (t == string(AnthropicToolTypeComputerToolset20260801) && AcceptsComputerToolset(g.caps)) {
		return t
	}
	generation := ComputerUseGeneration(g.caps)
	if base == "text_editor" {
		generation = TextEditorGeneration(g.caps)
	}
	if want, _ := NormalizedToolSpec(generation, base); want != "" {
		return want
	}
	return t
}

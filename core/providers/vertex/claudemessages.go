package vertex

import (
	"encoding/json"
	"net/url"
	"unsafe"

	"github.com/maximhq/bifrost/core/providers/anthropic"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/tidwall/gjson"
	"github.com/valyala/fasthttp"
)

// ClaudeMessagesSupport declares that Passthrough serves ClaudeMessages requests.
func (provider *VertexProvider) ClaudeMessagesSupport() bool {
	return true
}

// claudeMessagesURL returns the Anthropic publisher URL of a ClaudeMessages request (method is
// ":rawPredict" or ":streamRawPredict") for the resolved model, and its path; the client's path and
// query are ignored.
func (provider *VertexProvider) claudeMessagesURL(ctx *schemas.BifrostContext, key schemas.Key, req *schemas.BifrostPassthroughRequest, projectID, keyRegion, method string) (string, string) {
	// req.Model may be a routing name that key selection resolved through an alias; the
	// alias's ModelID is the Vertex model to call.
	model := req.Model
	if ra := schemas.GetResolvedAlias(ctx); ra != nil && ra.Config != nil && ra.Config.ModelID != "" {
		model = ra.Config.ModelID
	}
	requestURL := getVertexModelAwarePublisherModelURL(provider.networkConfig.BaseURL, keyRegion, "v1", projectID, "anthropic", model, method, resolveVertexForceSingleRegion(ctx, key), provider.logger)
	path := requestURL
	if u, err := url.Parse(requestURL); err == nil {
		path = u.Path
	}
	return requestURL, path
}

// setClaudeMessagesBody puts the Vertex form of a ClaudeMessages body (claudeMessagesBody) on
// fasthttpReq, after the headers, and returns the betas Vertex does not support. The body is never
// regex-rewritten.
func (provider *VertexProvider) setClaudeMessagesBody(fasthttpReq *fasthttp.Request, req *schemas.BifrostPassthroughRequest) []string {
	kept, dropped := anthropic.ClaudeMessagesBetas(req.SafeHeaders, provider.networkConfig.ExtraHeaders, schemas.Vertex, provider.networkConfig.BetaHeaderOverrides)
	// Vertex reads betas from the body only.
	fasthttpReq.Header.Del(anthropic.AnthropicBetaHeader)
	fasthttpReq.SetBodyRaw(claudeMessagesBody(req.Body, kept))
	return dropped
}

// claudeMessagesBody adapts an Anthropic Messages body to Vertex rawPredict: the top-level model
// is removed (the URL names it), anthropic_version is added when absent, and anthropic_beta is
// set to betas (removed when betas is empty; a client's own anthropic_beta member is replaced, so
// the filtered header betas are the only ones sent). Every other member is copied byte for byte,
// in one pass into one buffer. A body that is not a valid JSON object is returned unchanged for
// Vertex to refuse.
func claudeMessagesBody(body []byte, betas []string) []byte {
	raw := unsafe.String(unsafe.SliceData(body), len(body)) // read-only view, no copy
	parsed := gjson.Parse(raw)
	if !parsed.IsObject() || !gjson.Valid(raw) {
		return body
	}
	var betaJSON []byte
	if len(betas) > 0 {
		betaJSON, _ = json.Marshal(betas) // a []string always marshals
	}
	out := make([]byte, 0, len(body)+len(`,"anthropic_version":""`)+len(anthropic.VertexAnthropicVersion)+len(`,"anthropic_beta":`)+len(betaJSON))
	out = append(out, '{')
	hasVersion := false
	parsed.ForEach(func(k, v gjson.Result) bool {
		switch k.Str {
		case "model", "anthropic_beta":
			return true
		case "anthropic_version":
			hasVersion = true
		}
		if len(out) > 1 {
			out = append(out, ',')
		}
		// Parse records leading whitespace in Index, so ForEach offsets index body directly.
		out = append(out, body[k.Index:v.Index+len(v.Raw)]...)
		return true
	})
	if !hasVersion {
		if len(out) > 1 {
			out = append(out, ',')
		}
		out = append(out, `"anthropic_version":"`+anthropic.VertexAnthropicVersion+`"`...)
	}
	if betaJSON != nil {
		if len(out) > 1 {
			out = append(out, ',')
		}
		out = append(out, `"anthropic_beta":`...)
		out = append(out, betaJSON...)
	}
	return append(out, '}')
}

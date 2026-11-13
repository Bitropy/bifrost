package vertex

import (
	"encoding/json"
	"fmt"
	"net/url"
	"regexp"
	"unsafe"

	"github.com/maximhq/bifrost/core/providers/anthropic"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/tidwall/gjson"
	"github.com/valyala/fasthttp"
)

// claudeModelRe is what a Vertex model ID may contain. The model is spliced into the URL (the path on
// Messages, the host and location on count-tokens), and a passthrough policy only sees the route
// the client called, so "x/../../endpoints/1" or "m?key=1" would otherwise reach another Vertex
// resource with the key's token.
var claudeModelRe = regexp.MustCompile(`^[A-Za-z0-9._@:-]+$`)

// claudeRouteURL returns the Anthropic publisher URL of a ClaudeRoute request for the resolved model,
// and its path; the client's query is ignored. Messages calls the model with method (":rawPredict" or
// ":streamRawPredict"); count-tokens calls the shared count-tokens model on the model's host. A model
// that is not a single safe path segment is a 400.
func (provider *VertexProvider) claudeRouteURL(ctx *schemas.BifrostContext, key schemas.Key, req *schemas.BifrostPassthroughRequest, projectID, keyRegion, method string) (string, string, *schemas.BifrostError) {
	model := claudeRouteModel(ctx, req)
	if !claudeModelRe.MatchString(model) {
		return "", "", providerUtils.NewBifrostBadRequestError(fmt.Sprintf("Claude route passthrough: %q is not a Vertex model ID", model))
	}
	forceSingleRegion := resolveVertexForceSingleRegion(ctx, key)
	var requestURL string
	if req.ClaudeRoute == schemas.ClaudeRouteCountTokens {
		requestURL = getVertexAnthropicCountTokensURL(provider.networkConfig.BaseURL, keyRegion, projectID, model, forceSingleRegion, provider.logger)
	} else {
		requestURL = getVertexModelAwarePublisherModelURL(provider.networkConfig.BaseURL, keyRegion, "v1", projectID, "anthropic", model, method, forceSingleRegion, provider.logger)
	}
	path := requestURL
	if u, err := url.Parse(requestURL); err == nil {
		path = u.Path
	}
	return requestURL, path, nil
}

// claudeRouteModel is the Vertex model a ClaudeRoute request calls: req.Model may be a routing name
// that key selection resolved through an alias, whose ModelID is then the model.
func claudeRouteModel(ctx *schemas.BifrostContext, req *schemas.BifrostPassthroughRequest) string {
	if ra := schemas.GetResolvedAlias(ctx); ra != nil && ra.Config != nil && ra.Config.ModelID != "" {
		return ra.Config.ModelID
	}
	return req.Model
}

// setClaudeRouteBody puts the Vertex form of a ClaudeRoute body (claudeRouteBody) on fasthttpReq,
// after the headers, and returns the betas Vertex does not support. The body is never
// regex-rewritten.
func (provider *VertexProvider) setClaudeRouteBody(ctx *schemas.BifrostContext, fasthttpReq *fasthttp.Request, req *schemas.BifrostPassthroughRequest) []string {
	kept, dropped := anthropic.ClaudeRouteBetas(ctx, req.SafeHeaders, provider.networkConfig.ExtraHeaders, schemas.Vertex, provider.networkConfig.BetaHeaderOverrides)
	// Vertex reads betas from the body only.
	fasthttpReq.Header.Del(anthropic.AnthropicBetaHeader)
	// count-tokens reads the model from the body; rawPredict takes it from the URL.
	var bodyModel string
	if req.ClaudeRoute == schemas.ClaudeRouteCountTokens {
		bodyModel = claudeRouteModel(ctx, req)
	}
	fasthttpReq.SetBodyRaw(claudeRouteBody(req.Body, kept, bodyModel))
	return dropped
}

// claudeRouteBody adapts an Anthropic body to Vertex rawPredict. With bodyModel empty the top-level
// model is removed, since the URL names it. Otherwise model is set to bodyModel, in place or appended
// when absent: count-tokens reads the model from the body, so it must name the Vertex model the alias
// resolved to. anthropic_version is added when absent, and anthropic_beta is set to betas (removed when betas is empty; a client's own anthropic_beta member is replaced, so
// the filtered header betas are the only ones sent). Every other member is copied byte for byte,
// in one pass into one buffer. A body that is not a valid JSON object is returned unchanged for
// Vertex to refuse.
func claudeRouteBody(body []byte, betas []string, bodyModel string) []byte {
	raw := unsafe.String(unsafe.SliceData(body), len(body)) // read-only view, no copy
	parsed := gjson.Parse(raw)
	if !parsed.IsObject() || !gjson.Valid(raw) {
		return body
	}
	var betaJSON, modelJSON []byte
	if len(betas) > 0 {
		betaJSON, _ = json.Marshal(betas) // a []string always marshals
	}
	if bodyModel != "" {
		modelJSON, _ = json.Marshal(bodyModel) // a string always marshals
	}
	out := make([]byte, 0, len(body)+len(`,"model":`)+len(modelJSON)+len(`,"anthropic_version":""`)+len(anthropic.VertexAnthropicVersion)+len(`,"anthropic_beta":`)+len(betaJSON))
	out = append(out, '{')
	hasVersion, hasModel := false, false
	parsed.ForEach(func(k, v gjson.Result) bool {
		switch k.Str {
		case "model":
			if modelJSON == nil || hasModel {
				return true
			}
			hasModel = true
			if len(out) > 1 {
				out = append(out, ',')
			}
			out = append(out, `"model":`...)
			out = append(out, modelJSON...)
			return true
		case "anthropic_beta":
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
	if modelJSON != nil && !hasModel {
		if len(out) > 1 {
			out = append(out, ',')
		}
		out = append(out, `"model":`...)
		out = append(out, modelJSON...)
	}
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

// getVertexAnthropicCountTokensURL is the Anthropic count-tokens endpoint, served on the host and
// location model is served on.
func getVertexAnthropicCountTokensURL(base, region, projectID, model string, forceSingleRegion bool, logger schemas.Logger) string {
	baseURL := getVertexModelAwareAPIBaseURL(base, region, "v1", model, forceSingleRegion, logger)
	return fmt.Sprintf("%s/projects/%s/locations/%s/publishers/anthropic/models/count-tokens:rawPredict", baseURL, projectID, getVertexEffectiveRegion(region, model, forceSingleRegion))
}

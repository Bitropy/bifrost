package vertex

import (
	"encoding/json"
	"fmt"
	"net/url"
	"strings"
	"unsafe"

	"github.com/maximhq/bifrost/core/providers/anthropic"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
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

// passthroughURL returns the upstream URL of a generic passthrough request and, for a Google
// publisher path on an API key, the key query to append (empty means OAuth).
func (provider *VertexProvider) passthroughURL(key schemas.Key, req *schemas.BifrostPassthroughRequest, projectID, keyRegion string) (string, string, *schemas.BifrostError) {
	// Model calls stay on v1 as they always have; other resources (interactions) use the client's version.
	apiVersion := "v1"
	if req.APIVersion != "" && req.APIVersion != "v1beta" && !strings.Contains(req.Path, "/models/") {
		apiVersion = req.APIVersion
	}
	baseURL := getVertexAPIBaseURL(provider.networkConfig.BaseURL, keyRegion, apiVersion)

	// Normalize path: remove leading /v1 or /v1/ to avoid duplicate version segments (e.g. /v1/v1/...)
	path := req.Path
	for strings.HasPrefix(path, "/v1/") || path == "/v1" {
		path = strings.TrimPrefix(path, "/v1/")
		path = strings.TrimPrefix(path, "/v1")
		if path != "" && !strings.HasPrefix(path, "/") {
			path = "/" + path
		}
	}
	if rest, ok := cutVertexV1Beta1(path); ok {
		path, baseURL = rest, getVertexAPIBaseURL(provider.networkConfig.BaseURL, keyRegion, "v1beta1")
	}

	// Replace region in path with key's configured region (client path may have different region)
	if strings.Contains(path, "/locations/") {
		path = vertexLocationsPathRe.ReplaceAllString(path, "/locations/"+keyRegion)
		if strings.Contains(path, "/projects/") {
			path = vertexProjectsPathRe.ReplaceAllString(path, "/projects/"+projectID)
		}
	} else {
		// add projects/%s/locations/%s/publishers/google to path
		path = fmt.Sprintf("/projects/%s/locations/%s%s", projectID, keyRegion, path)
	}

	requestURL, err := providerUtils.BuildPassthroughURL(baseURL, path, req.RawQuery)
	if err != nil {
		return "", "", providerUtils.NewBifrostBadRequestError(err.Error())
	}

	// Only use API key for Google publisher endpoints; Anthropic/Mistral/OpenAPI-style paths require OAuth.
	authQuery := ""
	if key.Value.GetValue() != "" && strings.Contains(path, "publishers/google") {
		authQuery = fmt.Sprintf("key=%s", url.QueryEscape(key.Value.GetValue()))
	}
	return requestURL, authQuery, nil
}

// setPassthroughBody puts a generic passthrough body on fasthttpReq, after the headers, with the
// Vertex placeholders of a JSON body rewritten.
func setPassthroughBody(fasthttpReq *fasthttp.Request, req *schemas.BifrostPassthroughRequest, projectID, keyRegion string) {
	if len(req.Body) > 0 && strings.Contains(strings.ToLower(string(fasthttpReq.Header.ContentType())), "application/json") {
		// Replace placeholder project/location and expand short-form model names.
		if rewritten, changed := rewritePassthroughBody(req.Body, projectID, keyRegion); changed {
			fasthttpReq.SetBodyRaw(rewritten)
		} else {
			fasthttpReq.SetBody(req.Body)
		}
	} else if len(req.Body) > 0 {
		fasthttpReq.SetBody(req.Body)
	}
}

// passthroughWireFormat is the API req speaks: Anthropic Messages for a ClaudeMessages request,
// whatever its path, else the path's.
func passthroughWireFormat(req *schemas.BifrostPassthroughRequest) schemas.ModelProvider {
	if req.ClaudeMessages {
		return schemas.Anthropic
	}
	return PassthroughWireFormat(req.Path)
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

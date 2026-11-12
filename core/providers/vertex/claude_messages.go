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

// vertexAnthropicVersion is the anthropic_version Vertex requires in an Anthropic Messages body.
const vertexAnthropicVersion = "vertex-2023-10-16"

// ClaudeMessagesSupport declares that Passthrough serves ClaudeMessages requests.
func (provider *VertexProvider) ClaudeMessagesSupport() bool {
	return true
}

// passthroughURL returns the upstream URL of a passthrough request and, for a Google publisher
// path on an API key, the key query to append (empty means OAuth). A ClaudeMessages request goes
// to the Anthropic publisher's method (":rawPredict" or ":streamRawPredict") for the resolved
// model; the client's path and query are ignored.
func (provider *VertexProvider) passthroughURL(ctx *schemas.BifrostContext, key schemas.Key, req *schemas.BifrostPassthroughRequest, projectID, keyRegion, claudeMethod string) (string, string, *schemas.BifrostError) {
	if req.ClaudeMessages {
		// req.Model may be a routing name that key selection resolved through an alias; the
		// alias's ModelID is the Vertex model to call.
		model := req.Model
		if ra := schemas.GetResolvedAlias(ctx); ra != nil && ra.Config != nil && ra.Config.ModelID != "" {
			model = ra.Config.ModelID
		}
		return getVertexModelAwarePublisherModelURL(provider.networkConfig.BaseURL, keyRegion, "v1", projectID, "anthropic", model, claudeMethod, resolveVertexForceSingleRegion(ctx, key), provider.logger), "", nil
	}

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

// setPassthroughBody puts req's body on fasthttpReq, after the headers. A ClaudeMessages body is
// adapted by claudeMessagesBody and never regex-rewritten; it returns the betas Vertex dropped.
func (provider *VertexProvider) setPassthroughBody(fasthttpReq *fasthttp.Request, req *schemas.BifrostPassthroughRequest, projectID, keyRegion string) []string {
	if req.ClaudeMessages {
		kept, dropped := anthropic.ClaudeMessagesBetas(req.SafeHeaders, provider.networkConfig.ExtraHeaders, schemas.Vertex, provider.networkConfig.BetaHeaderOverrides)
		// Vertex reads betas from the body only.
		fasthttpReq.Header.Del(anthropic.AnthropicBetaHeader)
		fasthttpReq.SetBodyRaw(claudeMessagesBody(req.Body, kept))
		return dropped
	}
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
	return nil
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
	out := make([]byte, 0, len(body)+len(`,"anthropic_version":""`)+len(vertexAnthropicVersion)+len(`,"anthropic_beta":`)+len(betaJSON))
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
		out = append(out, `"anthropic_version":"`+vertexAnthropicVersion+`"`...)
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

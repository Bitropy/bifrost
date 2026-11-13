package bifrost

import (
	"fmt"
	"net/http"

	"github.com/maximhq/bifrost/core/schemas"
)

// passthroughPolicyRoute is the version and path a policy matches: a ClaudeRoute request is checked
// as the Anthropic route the client called, on every provider.
func passthroughPolicyRoute(req *schemas.BifrostPassthroughRequest) (apiVersion, path string) {
	if req.ClaudeRoute != "" {
		return "", req.ClaudeRoute.AnthropicPath()
	}
	return req.APIVersion, req.Path
}

// passthroughRouteAllowed reports whether a provider's passthrough policy forwards req.
// Host-backed routes (UpstreamURL) are not governed by the provider policy: they carry their
// own fixed route list and the caller's credential.
func passthroughRouteAllowed(policy *schemas.PassthroughPolicy, req *schemas.BifrostPassthroughRequest) bool {
	if policy == nil || req.UpstreamURL != "" {
		return true
	}
	apiVersion, path := passthroughPolicyRoute(req)
	for i := range policy.AllowedRoutes {
		if policy.AllowedRoutes[i].Matches(req.Method, apiVersion, path) {
			return true
		}
	}
	return false
}

// newPassthroughRouteNotAllowedError names the route as the client called it, so the
// operator can copy it into the provider's allowed routes.
func newPassthroughRouteNotAllowedError(provider schemas.ModelProvider, requestType schemas.RequestType, req *schemas.BifrostPassthroughRequest) schemas.BifrostError {
	apiVersion, routePath := passthroughPolicyRoute(req)
	if apiVersion != "" {
		routePath = "/" + apiVersion + routePath
	}
	return schemas.BifrostError{
		IsBifrostError: true,
		StatusCode:     new(http.StatusForbidden),
		Error: &schemas.ErrorField{
			Message: fmt.Sprintf("passthrough route %s %s is not allowed for provider %s; add it to the provider's passthrough allowed routes", req.Method, routePath, provider),
			Code:    new("passthrough_route_not_allowed"),
		},
		ExtraFields: schemas.BifrostErrorExtraFields{
			Provider:               provider,
			RequestType:            requestType,
			OriginalModelRequested: req.Model,
			ResolvedModelUsed:      req.Model,
		},
	}
}

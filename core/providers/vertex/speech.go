package vertex

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/core/providers/gemini"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	schemas "github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

// liveModelRefusalMessage is contract-pinned (BIT-3036 rule 7): bitrouter's
// failover.LocalRefusalMessages carries it verbatim so the router answers 400 with it
// instead of a redacted 500. Do not vary the wording.
const liveModelRefusalMessage = "live models require the Live API, which this integration does not serve"

// isVertexLiveModel reports whether the model name carries a "live" segment: the name is
// live, or "-live" appears at a segment boundary (suffix or embedded). Live models are
// served by the Live API only, which this integration does not speak.
func isVertexLiveModel(model string) bool {
	lower := strings.ToLower(model)
	return lower == "live" || strings.HasSuffix(lower, "-live") || strings.Contains(lower, "-live-")
}

// refuseLiveAudioModel answers every audio operation on a Live model with the canonical
// named refusal, before any HTTP call.
func refuseLiveAudioModel() *schemas.BifrostError {
	return providerUtils.NewBifrostOperationError(liveModelRefusalMessage, nil)
}

// audioGenerateContentURL builds the publisher-model URL for a Gemini-family audio call,
// with the chat path's project/region precedence and model-aware host selection.
func (provider *VertexProvider) audioGenerateContentURL(ctx *schemas.BifrostContext, key schemas.Key, model, method string) (string, *schemas.BifrostError) {
	projectID := resolveVertexProjectID(ctx, key)
	if projectID == "" {
		return "", providerUtils.NewConfigurationError("project ID is not set")
	}
	region := resolveVertexRegion(ctx, key)
	if region == "" {
		return "", providerUtils.NewConfigurationError("region is not set in key config")
	}
	return getVertexModelAwarePublisherModelURL(provider.networkConfig.BaseURL, region, "v1", projectID, "google", gemini.NormalizeModelName(model), method, resolveVertexForceSingleRegion(ctx, key), provider.logger), nil
}

// vertexAudioAuthQuery returns the ?key= query for API-key-only keys, or "" when the
// OAuth bearer applies. Chat's Gemini-family branch calls the API-key shortcut
// "key auth"; here it stays limited to keys that carry no service-account credential.
func vertexAudioAuthQuery(key schemas.Key) string {
	if vertexKeyUsesAPIKeyOnly(key) {
		return "key=" + url.QueryEscape(key.Value.GetValue())
	}
	return ""
}

// audioBearerToken mints the OAuth bearer for an audio call, or "" when the ?key= query
// applies. A failing token acquisition evicts the cached token source so the next
// attempt rebuilds it from scratch.
func (provider *VertexProvider) audioBearerToken(key schemas.Key) (string, *schemas.BifrostError) {
	if vertexKeyUsesAPIKeyOnly(key) {
		return "", nil
	}
	tokenSource, err := provider.getAuthTokenSource(key)
	if err != nil {
		return "", providerUtils.NewBifrostOperationError("error creating auth token source", err)
	}
	token, err := tokenSource.Token()
	if err != nil {
		provider.removeVertexClient(key)
		return "", providerUtils.NewBifrostOperationError("error getting token", err)
	}
	return "Bearer " + token.AccessToken, nil
}

// vertexAudioRequest posts a Gemini-family audio generateContent body and decodes the
// reply. Auth, URL resolution, credential eviction and error mapping match the chat
// path's Gemini-family branch; a large-response-mode reply returns no decoded body and
// no error, like the gemini provider's completeRequest.
func (provider *VertexProvider) vertexAudioRequest(ctx *schemas.BifrostContext, key schemas.Key, model, method string, jsonBody []byte) (*gemini.GenerateContentResponse, interface{}, time.Duration, map[string]string, *schemas.BifrostError) {
	completeURL, bifrostErr := provider.audioGenerateContentURL(ctx, key, model, method)
	if bifrostErr != nil {
		return nil, nil, 0, nil, bifrostErr
	}
	if query := vertexAudioAuthQuery(key); query != "" {
		completeURL += "?" + query
	}
	bearer, bifrostErr := provider.audioBearerToken(key)
	if bifrostErr != nil {
		return nil, nil, 0, nil, bifrostErr
	}

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	respOwned := true
	defer func() {
		if respOwned {
			fasthttp.ReleaseResponse(resp)
		}
	}()

	req.Header.SetMethod(http.MethodPost)
	req.Header.SetContentType("application/json")
	providerUtils.SetExtraHeaders(ctx, req, provider.networkConfig.ExtraHeaders, nil)
	if bearer != "" {
		req.Header.Set("Authorization", bearer)
	}
	req.SetRequestURI(completeURL)
	usedLargePayloadBody := providerUtils.ApplyLargePayloadRequestBody(ctx, req)
	if !usedLargePayloadBody {
		req.SetBody(jsonBody)
	}

	activeClient := providerUtils.PrepareResponseStreaming(ctx, provider.client, resp)
	latency, bifrostErr, wait := providerUtils.MakeRequestWithContext(ctx, activeClient, req, resp)
	defer wait()
	if bifrostErr != nil {
		return nil, nil, latency, nil, bifrostErr
	}
	if usedLargePayloadBody {
		providerUtils.DrainLargePayloadRemainder(ctx)
	}
	// Headers are extracted before the status check so error responses forward them too.
	providerResponseHeaders := providerUtils.ExtractProviderResponseHeaders(resp)

	if resp.StatusCode() != fasthttp.StatusOK {
		providerUtils.MaterializeStreamErrorBody(ctx, resp)
		if resp.StatusCode() == fasthttp.StatusUnauthorized || resp.StatusCode() == fasthttp.StatusForbidden {
			provider.removeVertexClient(key)
		}
		return nil, nil, latency, providerResponseHeaders, providerUtils.SetErrorLatency(parseVertexError(resp), latency)
	}

	responseBody, isLargeResp, decodeErr := providerUtils.FinalizeResponseWithLargeDetection(ctx, resp, provider.logger)
	if decodeErr != nil {
		return nil, nil, latency, providerResponseHeaders, decodeErr
	}
	if isLargeResp {
		respOwned = false
		return nil, nil, latency, providerResponseHeaders, nil
	}

	var geminiResponse gemini.GenerateContentResponse
	pt, ph := providerUtils.StartResponseParseSpan(ctx)
	umErr := sonic.Unmarshal(responseBody, &geminiResponse)
	if pt != nil {
		if umErr != nil {
			pt.EndSpan(ph, schemas.SpanStatusError, "response parse failed")
		} else {
			pt.EndSpan(ph, schemas.SpanStatusOk, "")
		}
	}
	if umErr != nil {
		return nil, nil, latency, providerResponseHeaders, providerUtils.NewBifrostOperationError(schemas.ErrProviderResponseUnmarshal, umErr)
	}

	var rawResponse interface{}
	if providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse) {
		if err := sonic.Unmarshal(responseBody, &rawResponse); err != nil {
			return nil, nil, latency, providerResponseHeaders, providerUtils.NewBifrostOperationError(schemas.ErrProviderResponseUnmarshal, err)
		}
	}
	return &geminiResponse, rawResponse, latency, providerResponseHeaders, nil
}

// decodeVertexAudioBase64 decodes Gemini inlineData payloads, which may arrive URL-safe
// and unpadded (same tolerance as the gemini provider's decoder).
func decodeVertexAudioBase64(b64 string) ([]byte, error) {
	standard := strings.NewReplacer("_", "/", "-", "+").Replace(b64)
	switch len(standard) % 4 {
	case 2:
		standard += "=="
	case 3:
		standard += "="
	}
	return base64.StdEncoding.DecodeString(standard)
}

// decodeVertexAudioStreamChunk decodes one SSE data line of a Vertex Gemini-family audio
// stream. An upstream error payload becomes a BifrostError carrying its status; a line
// that decodes as neither is returned as a plain error and skipped by the loops, exactly
// like the gemini provider's stream loop treats unparseable lines.
func decodeVertexAudioStreamChunk(jsonData []byte) (*gemini.GenerateContentResponse, *schemas.BifrostError, error) {
	if bytes.Contains(jsonData, []byte(`"error"`)) {
		var errorResp gemini.GeminiGenerationError
		if err := sonic.Unmarshal(jsonData, &errorResp); err == nil && errorResp.Error != nil {
			bifrostErr := &schemas.BifrostError{
				IsBifrostError: false,
				StatusCode:     schemas.Ptr(errorResp.Error.Code),
				Error: &schemas.ErrorField{
					Code:    schemas.Ptr(strconv.Itoa(errorResp.Error.Code)),
					Message: errorResp.Error.Message,
				},
			}
			if errorResp.Error.Status != "" {
				bifrostErr.Error.Type = schemas.Ptr(errorResp.Error.Status)
			}
			gemini.ApplyRetryInfo(bifrostErr, errorResp.Error.Details)
			// A payload without a usable code must not masquerade as a 0-status success;
			// ClassifyFailure has nothing to act on below a real status.
			if bifrostErr.StatusCode == nil || *bifrostErr.StatusCode == 0 {
				bifrostErr.StatusCode = schemas.Ptr(fasthttp.StatusBadGateway)
			}
			return nil, bifrostErr, nil
		}
	}
	var response gemini.GenerateContentResponse
	if err := sonic.Unmarshal(jsonData, &response); err != nil {
		return nil, nil, fmt.Errorf("failed to parse Gemini stream response: %w", err)
	}
	return &response, nil, nil
}

// Speech performs a text-to-speech request to a Gemini-family TTS model on Vertex AI,
// riding the generateContent endpoint with the gemini provider's speech converters.
func (provider *VertexProvider) Speech(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostSpeechRequest) (*schemas.BifrostSpeechResponse, *schemas.BifrostError) {
	if isVertexLiveModel(request.Model) {
		return nil, refuseLiveAudioModel()
	}
	jsonData, bifrostErr := providerUtils.CheckContextAndGetRequestBody(
		ctx,
		request,
		func() (providerUtils.RequestBodyWithExtraParams, error) {
			reqBody, err := gemini.ToGeminiSpeechRequest(request)
			if err != nil {
				return nil, err
			}
			// One body path with chat: the strip is a no-op for speech fields today, but
			// keeping it here means a future strip rule cannot silently skip audio.
			stripVertexGeminiUnsupportedFields(reqBody)
			return reqBody, nil
		})
	if bifrostErr != nil {
		return nil, bifrostErr
	}

	geminiResponse, rawResponse, latency, providerResponseHeaders, bifrostErr := provider.vertexAudioRequest(ctx, key, request.Model, ":generateContent", jsonData)
	if providerResponseHeaders != nil {
		ctx.SetValue(schemas.BifrostContextKeyProviderResponseHeaders, providerResponseHeaders)
	}
	if bifrostErr != nil {
		return nil, providerUtils.EnrichError(ctx, bifrostErr, jsonData, nil, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}

	// Large response mode: return lightweight response with metadata only.
	if isLargeResp, _ := ctx.Value(schemas.BifrostContextKeyLargeResponseMode).(bool); isLargeResp {
		return &schemas.BifrostSpeechResponse{
			ExtraFields: schemas.BifrostResponseExtraFields{
				Latency:                 latency.Milliseconds(),
				ProviderResponseHeaders: providerResponseHeaders,
			},
		}, nil
	}

	responseFormat := ""
	if request.Params != nil {
		responseFormat = request.Params.ResponseFormat
	}
	// ToBifrostSpeechResponse reads the format from the context to decide whether to wrap
	// the headerless PCM into WAV; it is always set so the lookup cannot fail empty.
	ctx.SetValue(gemini.BifrostContextKeyResponseFormat, responseFormat)
	response, convErr := geminiResponse.ToBifrostSpeechResponse(ctx)
	if convErr != nil {
		return nil, providerUtils.EnrichError(ctx, providerUtils.NewBifrostOperationError(schemas.ErrProviderResponseDecode, convErr), jsonData, nil, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}

	response.ExtraFields.Latency = latency.Milliseconds()
	response.ExtraFields.ProviderResponseHeaders = providerResponseHeaders

	if providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest) {
		providerUtils.ParseAndSetRawRequest(&response.ExtraFields, jsonData)
	}
	if providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse) {
		response.ExtraFields.RawResponse = rawResponse
	}
	return response, nil
}

// SpeechStream performs a streaming speech synthesis request to a Gemini-family TTS
// model on Vertex AI, mirroring the gemini provider's SSE loop over
// streamGenerateContent?alt=sse. Deltas are headerless PCM; the WAV wrap is the
// caller's (the unary path's) contract.
func (provider *VertexProvider) SpeechStream(ctx *schemas.BifrostContext, postHookRunner schemas.PostHookRunner, postHookSpanFinalizer func(context.Context), key schemas.Key, request *schemas.BifrostSpeechRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	if isVertexLiveModel(request.Model) {
		return nil, refuseLiveAudioModel()
	}
	jsonBody, bifrostErr := providerUtils.CheckContextAndGetRequestBody(
		ctx,
		request,
		func() (providerUtils.RequestBodyWithExtraParams, error) {
			reqBody, err := gemini.ToGeminiSpeechRequest(request)
			if err != nil {
				return nil, err
			}
			stripVertexGeminiUnsupportedFields(reqBody)
			return reqBody, nil
		})
	if bifrostErr != nil {
		return nil, bifrostErr
	}

	completeURL, bifrostErr := provider.audioGenerateContentURL(ctx, key, request.Model, ":streamGenerateContent")
	if bifrostErr != nil {
		return nil, bifrostErr
	}
	completeURL += "?alt=sse"
	if query := vertexAudioAuthQuery(key); query != "" {
		completeURL += "&" + query
	}
	bearer, bifrostErr := provider.audioBearerToken(key)
	if bifrostErr != nil {
		return nil, bifrostErr
	}

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	resp.StreamBody = true
	defer fasthttp.ReleaseRequest(req)

	req.Header.SetMethod(http.MethodPost)
	req.SetRequestURI(completeURL)
	req.Header.SetContentType("application/json")
	providerUtils.SetExtraHeaders(ctx, req, provider.networkConfig.ExtraHeaders, nil)
	if bearer != "" {
		req.Header.Set("Authorization", bearer)
	}
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Cache-Control", "no-cache")

	if !providerUtils.ApplyLargePayloadRequestBody(ctx, req) {
		req.SetBody(jsonBody)
	}

	startTime := time.Now()
	err := providerUtils.DoStreamingRequest(ctx, provider.streamingClient, req, resp)
	latency := time.Since(startTime)
	if err != nil {
		defer providerUtils.ReleaseStreamingResponse(ctx, resp)
		if errors.Is(err, context.Canceled) {
			return nil, providerUtils.EnrichError(ctx, &schemas.BifrostError{
				IsBifrostError: false,
				Error: &schemas.ErrorField{
					Type:    schemas.Ptr(schemas.RequestCancelled),
					Message: schemas.ErrRequestCancelled,
					Error:   err,
				},
			}, jsonBody, nil, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
		}
		if errors.Is(err, fasthttp.ErrTimeout) || errors.Is(err, context.DeadlineExceeded) {
			return nil, providerUtils.EnrichError(ctx, providerUtils.NewBifrostTimeoutError(schemas.ErrProviderRequestTimedOut, err), jsonBody, nil, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
		}
		// Request failed before the first response byte; surfaced as a retriable upstream
		// connection error so executeRequestWithRetries honors max_retries, as elsewhere.
		return nil, providerUtils.EnrichError(ctx, providerUtils.NewBifrostUpstreamConnectionError(schemas.ErrProviderDoRequest, err), jsonBody, nil, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}

	ctx.SetValue(schemas.BifrostContextKeyProviderResponseHeaders, providerUtils.ExtractProviderResponseHeaders(resp))
	if resp.StatusCode() != fasthttp.StatusOK {
		defer providerUtils.ReleaseStreamingResponse(ctx, resp)
		if resp.StatusCode() == fasthttp.StatusUnauthorized || resp.StatusCode() == fasthttp.StatusForbidden {
			provider.removeVertexClient(key)
		}
		return nil, providerUtils.EnrichError(ctx, parseVertexError(resp), jsonBody, nil, provider.sendBackRawRequest, provider.sendBackRawResponse, latency)
	}

	providerUtils.SetStreamIdleTimeoutIfEmpty(ctx, provider.networkConfig.StreamIdleTimeoutInSeconds)

	if providerUtils.SetupStreamingPassthrough(ctx, resp) {
		responseChan := make(chan *schemas.BifrostStreamChunk)
		providerUtils.CloseStream(ctx, responseChan)
		return responseChan, nil
	}

	responseChan := make(chan *schemas.BifrostStreamChunk, schemas.DefaultStreamBufferSize)

	go func() {
		defer providerUtils.EnsureStreamFinalizerCalled(ctx, postHookSpanFinalizer)
		defer func() {
			if ctx.Err() == context.Canceled {
				providerUtils.HandleStreamCancellation(ctx, postHookRunner, responseChan, provider.logger, postHookSpanFinalizer, jsonBody)
			} else if ctx.Err() == context.DeadlineExceeded {
				providerUtils.HandleStreamTimeout(ctx, postHookRunner, responseChan, provider.logger, postHookSpanFinalizer, jsonBody)
			}
			providerUtils.CloseStream(ctx, responseChan)
		}()
		defer providerUtils.ReleaseStreamingResponse(ctx, resp)

		reader, releaseGzip := providerUtils.DecompressStreamBody(resp)
		defer releaseGzip()
		reader, stopIdleTimeout := providerUtils.NewIdleTimeoutReader(reader, resp.BodyStream(), providerUtils.GetStreamIdleTimeout(ctx), ctx)
		defer stopIdleTimeout()
		stopCancellation := providerUtils.SetupStreamCancellation(ctx, resp.BodyStream(), provider.logger)
		defer stopCancellation()

		sseReader := providerUtils.GetSSEDataReader(ctx, reader)
		chunkIndex := -1
		usage := &schemas.SpeechUsage{}
		lastChunkTime := startTime

		for {
			if ctx.Err() != nil {
				return
			}
			data, readErr := sseReader.ReadDataLine()
			if readErr != nil {
				if ctx.Err() != nil {
					return
				}
				if readErr != io.EOF {
					ctx.SetValue(schemas.BifrostContextKeyStreamEndIndicator, true)
					provider.logger.Warn("Error reading stream: %v", readErr)
					providerUtils.ProcessAndSendError(ctx, postHookRunner, readErr, responseChan, provider.logger, postHookSpanFinalizer)
					return
				}
				break
			}

			parseStart := time.Now()
			geminiResponse, apiErr, decodeErr := decodeVertexAudioStreamChunk(data)
			schemas.AddStreamParse(ctx, time.Since(parseStart))
			if apiErr != nil {
				ctx.SetValue(schemas.BifrostContextKeyStreamEndIndicator, true)
				providerUtils.ProcessAndSendBifrostError(ctx, postHookRunner, apiErr, responseChan, provider.logger, postHookSpanFinalizer)
				return
			}
			if decodeErr != nil {
				provider.logger.Warn("Failed to process chunk: %v", decodeErr)
				continue
			}

			var audioChunk []byte
			if len(geminiResponse.Candidates) > 0 && geminiResponse.Candidates[0].Content != nil {
				for _, part := range geminiResponse.Candidates[0].Content.Parts {
					if part.InlineData != nil && len(part.InlineData.Data) > 0 {
						decoded, err := decodeVertexAudioBase64(part.InlineData.Data)
						if err != nil {
							provider.logger.Warn("Failed to decode base64 audio data: %v", err)
							continue
						}
						audioChunk = append(audioChunk, decoded...)
					}
				}
			}

			if len(geminiResponse.Candidates) > 0 && (geminiResponse.Candidates[0].FinishReason != "" || geminiResponse.UsageMetadata != nil) && geminiResponse.UsageMetadata != nil {
				usage.InputTokens = int(geminiResponse.UsageMetadata.PromptTokenCount)
				usage.OutputTokens = int(geminiResponse.UsageMetadata.CandidatesTokenCount)
				usage.TotalTokens = int(geminiResponse.UsageMetadata.TotalTokenCount)
			}

			if len(audioChunk) > 0 {
				chunkIndex++
				response := &schemas.BifrostSpeechStreamResponse{
					Type:  schemas.SpeechStreamResponseTypeDelta,
					Audio: audioChunk,
					ExtraFields: schemas.BifrostResponseExtraFields{
						ChunkIndex: chunkIndex,
						Latency:    time.Since(lastChunkTime).Milliseconds(),
					},
				}
				lastChunkTime = time.Now()
				if providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse) {
					response.ExtraFields.RawResponse = data
				}
				providerUtils.ProcessAndSendResponse(ctx, postHookRunner, providerUtils.GetBifrostResponseForStreamResponse(nil, nil, nil, response, nil, nil), responseChan, postHookSpanFinalizer)
			}
		}
		response := &schemas.BifrostSpeechStreamResponse{
			Type:  schemas.SpeechStreamResponseTypeDone,
			Usage: usage,
			ExtraFields: schemas.BifrostResponseExtraFields{
				ChunkIndex: chunkIndex + 1,
				Latency:    time.Since(startTime).Milliseconds(),
			},
		}
		response.BackfillParams(request)
		if providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest) {
			providerUtils.ParseAndSetRawRequest(&response.ExtraFields, jsonBody)
		}
		ctx.SetValue(schemas.BifrostContextKeyStreamEndIndicator, true)
		providerUtils.ProcessAndSendResponse(ctx, postHookRunner, providerUtils.GetBifrostResponseForStreamResponse(nil, nil, nil, response, nil, nil), responseChan, postHookSpanFinalizer)
	}()

	return responseChan, nil
}

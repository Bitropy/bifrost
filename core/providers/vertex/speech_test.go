package vertex

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
	"unicode/utf8"

	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/tidwall/gjson"
	"golang.org/x/oauth2"
)

// vertexLiveModelRefusal is contract-pinned: bitrouter's failover.LocalRefusalMessages
// carries it verbatim so the router answers 400 with it instead of a redacted 500.
const vertexLiveModelRefusal = "live models require the Live API, which this integration does not serve"

// vertexAudioTestProvider points every AI Platform URL at the test server and installs a
// static OAuth token source, so no real Google endpoint or credential is touched.
func vertexAudioTestProvider(t testing.TB, handler http.Handler) (*VertexProvider, schemas.Key) {
	t.Helper()
	provider, key, _ := vertexAudioTestProviderURL(t, handler)
	return provider, key
}

// vertexAudioTestProviderURL is vertexAudioTestProvider that also hands back the test
// server's URL, for tests that mint OAuth tokens through it.
func vertexAudioTestProviderURL(t testing.TB, handler http.Handler) (*VertexProvider, schemas.Key, string) {
	t.Helper()
	upstream := httptest.NewServer(handler)
	t.Cleanup(upstream.Close)
	config := &schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{
		BaseURL: upstream.URL, AllowPrivateNetwork: true, DefaultRequestTimeoutInSeconds: 5,
	}}
	config.CheckAndSetDefaults()
	provider, err := NewVertexProvider(config, vertexEndpointTestLogger{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(provider.client.CloseIdleConnections)
	t.Cleanup(provider.streamingClient.CloseIdleConnections)
	key := schemas.Key{VertexKeyConfig: &schemas.VertexKeyConfig{
		ProjectID:       *schemas.NewSecretVar("project-123"),
		Region:          *schemas.NewSecretVar("us-central1"),
		AuthCredentials: *schemas.NewSecretVar("test-credentials"),
	}}
	provider.tokenSources.Store(getClientKey(vertexCredentialIdentity(key)), oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "test-token"}))
	return provider, key, upstream.URL
}

// vertexAudioAPIKey swaps the key to API-key-only auth: the ?key= query shortcut, no bearer.
// The key config is cloned so callers sharing one key do not see each other's edits.
func vertexAudioAPIKey(key schemas.Key) schemas.Key {
	cfg := *key.VertexKeyConfig
	cfg.AuthCredentials = *schemas.NewSecretVar("")
	key.VertexKeyConfig = &cfg
	key.Value = *schemas.NewSecretVar("test-api-key")
	return key
}

// noAudioHooks is a post-hook runner that leaves every chunk untouched.
func noAudioHooks(_ *schemas.BifrostContext, resp *schemas.BifrostResponse, err *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError) {
	return resp, err
}

// vertexAudioSSE frames the given payloads as data-only SSE lines, the shape
// streamGenerateContent?alt=sse answers with.
func vertexAudioSSE(payloads ...string) string {
	var b bytes.Buffer
	for _, p := range payloads {
		b.WriteString("data: " + p + "\n\n")
	}
	return b.String()
}

func TestVertexAudioLiveModelPredicate(t *testing.T) {
	t.Parallel()
	for _, tt := range []struct {
		model string
		live  bool
	}{
		{"gemini-3.5-transcribe-live", true},
		{"gemini-2.5-flash-preview-tts-live", true},
		{"live", true},
		{"gemini-live-2", true},
		{"olive-tts", false},
		{"alive-1", false},
		{"gemini-2.5-flash-preview-tts", false},
		{"gemini-3.5-transcribe", false},
	} {
		if got := isVertexLiveModel(tt.model); got != tt.live {
			t.Errorf("isVertexLiveModel(%q) = %v, want %v", tt.model, got, tt.live)
		}
	}
}

// TestVertexAudioLiveModelRefusedBeforeHTTP pins the canonical refusal string (rule 7)
// on all four audio methods and that none of them dials upstream.
func TestVertexAudioLiveModelRefusedBeforeHTTP(t *testing.T) {
	requests := 0
	provider, key := vertexAudioTestProvider(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests++
		_, _ = io.WriteString(w, `{}`)
	}))
	speech := &schemas.BifrostSpeechRequest{Provider: schemas.Vertex, Model: "gemini-3.5-transcribe-live", Input: &schemas.SpeechInput{Input: "hi"}}
	transcription := &schemas.BifrostTranscriptionRequest{Provider: schemas.Vertex, Model: "gemini-3.5-transcribe-live", Input: &schemas.TranscriptionInput{File: []byte("RIFF____WAVE")}}
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	type refusal interface{ GetErrorString() string }
	calls := []func() refusal{
		func() refusal {
			_, err := provider.Speech(ctx, key, speech)
			return err
		},
		func() refusal {
			_, err := provider.SpeechStream(ctx, noAudioHooks, nil, key, speech)
			return err
		},
		func() refusal {
			_, err := provider.Transcription(ctx, key, transcription)
			return err
		},
		func() refusal {
			_, err := provider.TranscriptionStream(ctx, noAudioHooks, nil, key, transcription)
			return err
		},
	}
	for i, call := range calls {
		err := call()
		if err == nil {
			t.Fatalf("call %d: live model not refused", i)
		}
		if got := err.GetErrorString(); got != vertexLiveModelRefusal {
			t.Fatalf("call %d: refusal = %q, want the canonical string %q", i, got, vertexLiveModelRefusal)
		}
	}
	if requests != 0 {
		t.Fatalf("live model reached upstream %d times", requests)
	}
}

func TestVertexSpeechRequestBuild(t *testing.T) {
	type sent struct{ path, query, auth, body string }
	requests := make(chan sent, 4)
	provider, key := vertexAudioTestProvider(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		requests <- sent{r.URL.Path, r.URL.RawQuery, r.Header.Get("Authorization"), string(data)}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"audio/L16;codec=pcm;rate=24000","data":"cGNt"}}]}}]}`)
	}))
	voice := "Kore"
	request := func(key schemas.Key, region string) *sent {
		cfg := *key.VertexKeyConfig
		cfg.Region = *schemas.NewSecretVar(region)
		key.VertexKeyConfig = &cfg
		ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
		_, bifrostErr := provider.Speech(ctx, key, &schemas.BifrostSpeechRequest{
			Provider: schemas.Vertex, Model: "gemini-2.5-flash-preview-tts",
			Input:  &schemas.SpeechInput{Input: "say chevre"},
			Params: &schemas.SpeechParameters{ResponseFormat: "wav", VoiceConfig: &schemas.SpeechVoiceInput{Voice: &voice}},
		})
		if bifrostErr != nil {
			t.Fatalf("Speech: %s", bifrostErr.GetErrorString())
		}
		select {
		case got := <-requests:
			return &got
		default:
			t.Fatal("no request reached the base")
			return nil
		}
	}

	for _, tt := range []struct {
		name   string
		key    func(schemas.Key) schemas.Key
		region string
		path   string
		query  string
		auth   string
	}{
		{"regional bearer", func(k schemas.Key) schemas.Key { return k }, "us-central1",
			"/v1/projects/project-123/locations/us-central1/publishers/google/models/gemini-2.5-flash-preview-tts:generateContent", "", "Bearer test-token"},
		{"api key query", vertexAudioAPIKey, "us-central1",
			"/v1/projects/project-123/locations/us-central1/publishers/google/models/gemini-2.5-flash-preview-tts:generateContent", "key=test-api-key", ""},
		{"global region", func(k schemas.Key) schemas.Key { return k }, "global",
			"/v1/projects/project-123/locations/global/publishers/google/models/gemini-2.5-flash-preview-tts:generateContent", "", "Bearer test-token"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got := request(tt.key(key), tt.region)
			if got.path != tt.path || got.query != tt.query || got.auth != tt.auth {
				t.Fatalf("upstream = %s ?%s (%s), want %s ?%s (%s)", got.path, got.query, got.auth, tt.path, tt.query, tt.auth)
			}
			// The generateContent TTS shape: text part, AUDIO-only response modalities and
			// the prebuilt voice — and stripVertexGeminiUnsupportedFields must leave them be.
			if got := gjson.Get(got.body, "contents.0.parts.0.text").String(); got != "say chevre" {
				t.Errorf("input text = %q", got)
			}
			if got := gjson.Get(got.body, "generationConfig.responseModalities").Array(); len(got) != 1 || got[0].String() != "AUDIO" {
				t.Errorf("responseModalities = %v, want [AUDIO]", got)
			}
			if got := gjson.Get(got.body, "generationConfig.speechConfig.voice_config.prebuiltVoiceConfig.voiceName").String(); got != "Kore" {
				t.Errorf("voiceName = %q, want Kore", got)
			}
		})
	}
}

func TestVertexSpeechURLHostSelection(t *testing.T) {
	// Flag one model multi-region-only so the pool promotion is exercised: audio rides
	// the same model-aware host selection as partner-model chat.
	providerUtils.SetCapabilityResolver(func(provider schemas.ModelProvider, model string) *schemas.ModelCapabilities {
		if provider == schemas.Vertex && model == "gemini-tts-multi-region" {
			return &schemas.ModelCapabilities{IsVertexMultiRegionOnly: schemas.Ptr(true)}
		}
		return nil
	})
	t.Cleanup(func() { providerUtils.SetCapabilityResolver(nil) })

	config := &schemas.ProviderConfig{NetworkConfig: schemas.NetworkConfig{DefaultRequestTimeoutInSeconds: 5}}
	config.CheckAndSetDefaults()
	provider, err := NewVertexProvider(config, vertexEndpointTestLogger{})
	if err != nil {
		t.Fatal(err)
	}
	key := schemas.Key{VertexKeyConfig: &schemas.VertexKeyConfig{
		ProjectID: *schemas.NewSecretVar("p"), Region: *schemas.NewSecretVar("us-central1"),
	}}
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	urlFor := func(region, model string) string {
		key.VertexKeyConfig.Region = *schemas.NewSecretVar(region)
		got, bifrostErr := provider.audioGenerateContentURL(ctx, key, model, ":generateContent")
		if bifrostErr != nil {
			t.Fatalf("audioGenerateContentURL(%s): %s", region, bifrostErr.GetErrorString())
		}
		return got
	}
	for _, tt := range []struct{ region, model, want string }{
		{"us-central1", "gemini-2.5-flash-preview-tts", "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/publishers/google/models/gemini-2.5-flash-preview-tts:generateContent"},
		{"global", "gemini-2.5-flash-preview-tts", "https://aiplatform.googleapis.com/v1/projects/p/locations/global/publishers/google/models/gemini-2.5-flash-preview-tts:generateContent"},
		// The capability resolver flags this model multi-region-only (see helper): audio
		// rides the model-aware builder, so a us-* region is promoted to the pool host.
		{"us-central1", "gemini-tts-multi-region", "https://aiplatform.us.rep.googleapis.com/v1/projects/p/locations/us/publishers/google/models/gemini-tts-multi-region:generateContent"},
		{"us-central1", "google/gemini-2.5-flash-preview-tts", "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/publishers/google/models/gemini-2.5-flash-preview-tts:generateContent"},
	} {
		if got := urlFor(tt.region, tt.model); got != tt.want {
			t.Errorf("url(%s, %s) = %q, want %q", tt.region, tt.model, got, tt.want)
		}
	}
}

func TestVertexSpeechResponseConversion(t *testing.T) {
	speech := func(t *testing.T, format string, body string) *schemas.BifrostSpeechResponse {
		t.Helper()
		provider, key := vertexAudioTestProvider(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, body)
		}))
		ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
		request := &schemas.BifrostSpeechRequest{
			Provider: schemas.Vertex, Model: "gemini-2.5-flash-preview-tts",
			Input:  &schemas.SpeechInput{Input: "héllo"},
			Params: &schemas.SpeechParameters{ResponseFormat: format},
		}
		response, bifrostErr := provider.Speech(ctx, key, request)
		if bifrostErr != nil {
			t.Fatalf("Speech: %s", bifrostErr.GetErrorString())
		}
		response.BackfillParams(request)
		return response
	}

	pcmA, pcmB := []byte("pcm-frame-a"), []byte("pcm-frame-b")
	b64A, b64B := base64.StdEncoding.EncodeToString(pcmA), base64.StdEncoding.EncodeToString(pcmB)
	const usageMetadata = `"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":7,"totalTokenCount":12}`

	t.Run("wav wraps pcm with a riff header", func(t *testing.T) {
		response := speech(t, "wav", `{"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"audio/L16;codec=pcm;rate=24000","data":"`+b64A+`"}}]},"finishReason":"STOP"}],`+usageMetadata+`}`)
		if !bytes.HasPrefix(response.Audio, []byte("RIFF")) || !bytes.Equal(response.Audio[8:12], []byte("WAVE")) {
			t.Fatalf("audio = %q, want a RIFF/WAVE header", response.Audio)
		}
		if !bytes.HasSuffix(response.Audio, pcmA) {
			t.Fatalf("audio payload = %q, want the PCM bytes appended", response.Audio)
		}
		if response.Usage == nil || response.Usage.InputTokens != 5 || response.Usage.OutputTokens != 7 || response.Usage.TotalTokens != 12 {
			t.Fatalf("usage = %+v, want token counts 5/7/12", response.Usage)
		}
		if response.Usage.InputChars != utf8.RuneCountInString("héllo") {
			t.Fatalf("input chars = %d, want the rune count of the input", response.Usage.InputChars)
		}
	})

	t.Run("empty format passes raw pcm through", func(t *testing.T) {
		response := speech(t, "", `{"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"audio/L16;codec=pcm;rate=24000","data":"`+b64A+`"}}]},"finishReason":"STOP"}],`+usageMetadata+`}`)
		if !bytes.Equal(response.Audio, pcmA) {
			t.Fatalf("audio = %q, want the raw PCM bytes %q", response.Audio, pcmA)
		}
	})

	t.Run("multi-part audio concatenates", func(t *testing.T) {
		response := speech(t, "", `{"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"audio/L16","data":"`+b64A+`"}},{"inlineData":{"mimeType":"audio/L16","data":"`+b64B+`"}}]}}],`+usageMetadata+`}`)
		if want := append(append([]byte{}, pcmA...), pcmB...); !bytes.Equal(response.Audio, want) {
			t.Fatalf("audio = %q, want %q", response.Audio, want)
		}
	})
}

func TestVertexSpeechErrorMapping(t *testing.T) {
	provider, key := vertexAudioTestProvider(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = io.WriteString(w, `{"error":{"code":404,"message":"model gemini-x-tts is not supported","status":"NOT_FOUND"}}`)
	}))
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	_, bifrostErr := provider.Speech(ctx, key, &schemas.BifrostSpeechRequest{
		Provider: schemas.Vertex, Model: "gemini-x-tts", Input: &schemas.SpeechInput{Input: "hi"},
	})
	if bifrostErr == nil {
		t.Fatal("404 not surfaced as an error")
	}
	if bifrostErr.StatusCode == nil || *bifrostErr.StatusCode != http.StatusNotFound {
		t.Fatalf("status = %v, want 404", bifrostErr.StatusCode)
	}
	if bifrostErr.Error == nil || !bytes.Contains([]byte(bifrostErr.Error.Message), []byte("not supported")) {
		t.Fatalf("message = %v, want the upstream text", bifrostErr.Error)
	}
}

func TestVertexSpeechStream(t *testing.T) {
	b64 := func(b string) string { return base64.StdEncoding.EncodeToString([]byte(b)) }
	sse := vertexAudioSSE(
		`{"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"audio/L16;codec=pcm;rate=24000","data":"`+b64("abc")+`"}}]}}]}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"audio/L16;codec=pcm;rate=24000","data":"`+b64("def")+`"}}]}}]}`,
		`{"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":7,"totalTokenCount":12}}`,
	)
	provider, key := vertexAudioTestProvider(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, sse)
	}))
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	request := &schemas.BifrostSpeechRequest{
		Provider: schemas.Vertex, Model: "gemini-2.5-flash-preview-tts",
		Input: &schemas.SpeechInput{Input: "héllo"},
	}
	stream, bifrostErr := provider.SpeechStream(ctx, noAudioHooks, nil, key, request)
	if bifrostErr != nil {
		t.Fatalf("SpeechStream: %s", bifrostErr.GetErrorString())
	}
	var deltas [][]byte
	var done *schemas.BifrostSpeechStreamResponse
	timeout := time.After(5 * time.Second)
	for done == nil {
		select {
		case chunk, ok := <-stream:
			if !ok {
				t.Fatalf("stream closed after %d deltas without a done chunk", len(deltas))
			}
			if chunk.BifrostError != nil {
				t.Fatalf("stream chunk error: %s", chunk.BifrostError.GetErrorString())
			}
			response := chunk.BifrostSpeechStreamResponse
			if response.Type == schemas.SpeechStreamResponseTypeDone {
				done = response
			} else {
				deltas = append(deltas, response.Audio)
			}
		case <-timeout:
			t.Fatal("stream did not finish")
		}
	}
	if len(deltas) != 2 || string(deltas[0]) != "abc" || string(deltas[1]) != "def" {
		t.Fatalf("deltas = %q, want [abc def]", deltas)
	}
	if done.Usage == nil || done.Usage.InputTokens != 5 || done.Usage.OutputTokens != 7 || done.Usage.TotalTokens != 12 {
		t.Fatalf("done usage = %+v, want token counts 5/7/12", done.Usage)
	}
	// BackfillParams on the done chunk carries the billable character count.
	if done.Usage.InputChars != utf8.RuneCountInString("héllo") {
		t.Fatalf("done input chars = %d, want the rune count of the input", done.Usage.InputChars)
	}
}

// TestVertexSpeechAuthErrorEvictsAndReacquiresToken pins the eviction clause of the audio
// auth path: a 401 removes the cached token source (the chat path's established pin — see
// TestVertexEndpointErrorEvictsCredentials), and the next request re-acquires a token from
// the credential's token_uri instead of reusing the evicted one.
func TestVertexSpeechAuthErrorEvictsAndReacquiresToken(t *testing.T) {
	var generateCalls int32
	bearers := make(chan string, 4)
	provider, key, srvURL := vertexAudioTestProviderURL(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/token" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"access_token":"reacquired-token","token_type":"Bearer","expires_in":3600}`)
			return
		}
		bearers <- r.Header.Get("Authorization")
		if n := atomic.AddInt32(&generateCalls, 1); n == 1 || n == 3 {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = io.WriteString(w, `{"error":{"code":401,"message":"invalid token","status":"UNAUTHENTICATED"}}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"audio/L16","data":"cGNt"}}]}}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`)
	}))
	// The key authenticates through this server's token endpoint; a stale cached source
	// stands in for a token that upstream has since rejected.
	creds, _ := json.Marshal(map[string]string{
		"type": "authorized_user", "client_id": "c", "client_secret": "s", "refresh_token": "r", "token_uri": srvURL + "/token",
	})
	key.VertexKeyConfig.AuthCredentials = *schemas.NewSecretVar(string(creds))
	clientKey := getClientKey(vertexCredentialIdentity(key))
	provider.tokenSources.Store(clientKey, oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "stale-token"}))

	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	request := &schemas.BifrostSpeechRequest{
		Provider: schemas.Vertex, Model: "gemini-2.5-flash-preview-tts",
		Input: &schemas.SpeechInput{Input: "hi"},
	}

	// First call carries the stale token and is rejected with 401, which must evict it.
	if _, bifrostErr := provider.Speech(ctx, key, request); bifrostErr == nil || bifrostErr.StatusCode == nil || *bifrostErr.StatusCode != http.StatusUnauthorized {
		t.Fatalf("first Speech error = %v, want a 401", bifrostErr)
	}
	if got := <-bearers; got != "Bearer stale-token" {
		t.Fatalf("first call sent %q, want the stale bearer", got)
	}
	if _, ok := provider.tokenSources.Load(clientKey); ok {
		t.Fatal("cached token source was not evicted after 401")
	}

	// Second call re-acquires through the credential's token_uri.
	if _, bifrostErr := provider.Speech(ctx, key, request); bifrostErr != nil {
		t.Fatalf("second Speech after eviction: %s", bifrostErr.GetErrorString())
	}
	if got := <-bearers; got != "Bearer reacquired-token" {
		t.Fatalf("second call sent %q, want the re-acquired bearer", got)
	}

	// The stream path shares the eviction: another 401 clears the re-acquired source.
	if _, bifrostErr := provider.SpeechStream(ctx, noAudioHooks, nil, key, request); bifrostErr == nil || bifrostErr.StatusCode == nil || *bifrostErr.StatusCode != http.StatusUnauthorized {
		t.Fatalf("SpeechStream error = %v, want a 401", bifrostErr)
	}
	if got := <-bearers; got != "Bearer reacquired-token" {
		t.Fatalf("stream call sent %q, want the re-acquired bearer", got)
	}
	if _, ok := provider.tokenSources.Load(clientKey); ok {
		t.Fatal("cached token source was not evicted after stream 401")
	}
}

// TestVertexSpeechStreamAPIKeyQuery pins that an API-key-only key authenticates the SSE
// call through the key query alongside alt=sse, with no bearer header.
func TestVertexSpeechStreamAPIKeyQuery(t *testing.T) {
	type seen struct{ path, query, auth string }
	requests := make(chan seen, 1)
	sse := vertexAudioSSE(
		`{"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"audio/L16","data":"YWJj"}}]}}]}`,
		`{"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`,
	)
	provider, key := vertexAudioTestProvider(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests <- seen{r.URL.Path, r.URL.RawQuery, r.Header.Get("Authorization")}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, sse)
	}))
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	stream, bifrostErr := provider.SpeechStream(ctx, noAudioHooks, nil, vertexAudioAPIKey(key), &schemas.BifrostSpeechRequest{
		Provider: schemas.Vertex, Model: "gemini-2.5-flash-preview-tts",
		Input: &schemas.SpeechInput{Input: "hi"},
	})
	if bifrostErr != nil {
		t.Fatalf("SpeechStream: %s", bifrostErr.GetErrorString())
	}
	timeout := time.After(5 * time.Second)
	for open := true; open; {
		select {
		case chunk, ok := <-stream:
			if !ok {
				open = false
				break
			}
			if chunk.BifrostError != nil {
				t.Fatalf("stream chunk error: %s", chunk.BifrostError.GetErrorString())
			}
		case <-timeout:
			t.Fatal("stream did not finish")
		}
	}
	select {
	case got := <-requests:
		if want := "/v1/projects/project-123/locations/us-central1/publishers/google/models/gemini-2.5-flash-preview-tts:streamGenerateContent"; got.path != want {
			t.Fatalf("path = %s, want %s", got.path, want)
		}
		if got.query != "alt=sse&key=test-api-key" {
			t.Fatalf("query = %q, want the api key alongside alt=sse", got.query)
		}
		if got.auth != "" {
			t.Fatalf("authorization = %q, want none for an api-key-only key", got.auth)
		}
	default:
		t.Fatal("no request reached the base")
	}
}

// TestVertexSpeechKeyWithValueAndCredentialsUsesBearer pins the auth gating chat diverges
// on: a key carrying both an API-key Value and service-account credentials must use the
// OAuth bearer and must not append ?key=.
func TestVertexSpeechKeyWithValueAndCredentialsUsesBearer(t *testing.T) {
	type seen struct{ query, auth string }
	requests := make(chan seen, 1)
	provider, key := vertexAudioTestProvider(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests <- seen{r.URL.RawQuery, r.Header.Get("Authorization")}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"audio/L16","data":"cGNt"}}]}}]}`)
	}))
	both := key
	both.Value = *schemas.NewSecretVar("api-key-must-not-be-used")
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	if _, bifrostErr := provider.Speech(ctx, both, &schemas.BifrostSpeechRequest{
		Provider: schemas.Vertex, Model: "gemini-2.5-flash-preview-tts",
		Input: &schemas.SpeechInput{Input: "hi"},
	}); bifrostErr != nil {
		t.Fatalf("Speech: %s", bifrostErr.GetErrorString())
	}
	select {
	case got := <-requests:
		if got.query != "" {
			t.Fatalf("query = %q, want none: credentials must win over the api key", got.query)
		}
		if got.auth != "Bearer test-token" {
			t.Fatalf("authorization = %q, want the OAuth bearer", got.auth)
		}
	default:
		t.Fatal("no request reached the base")
	}
}

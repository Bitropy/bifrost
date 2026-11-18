package vertex

import (
	"context"
	"encoding/base64"
	"io"
	"net/http"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/tidwall/gjson"
)

func TestVertexTranscriptionRequestBuild(t *testing.T) {
	type sent struct{ path, query, auth, body string }
	requests := make(chan sent, 3)
	provider, key := vertexAudioTestProvider(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		requests <- sent{r.URL.Path, r.URL.RawQuery, r.Header.Get("Authorization"), string(data)}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]}}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`)
	}))
	// RIFF....WAVE magic so DetectAudioMimeType answers audio/wav.
	wav := append([]byte("RIFF\x00\x00\x00\x00WAVE"), []byte("fmt data")...)
	transcribe := func(key schemas.Key, prompt *string) sent {
		ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
		_, bifrostErr := provider.Transcription(ctx, key, &schemas.BifrostTranscriptionRequest{
			Provider: schemas.Vertex, Model: "gemini-3.5-transcribe",
			Input:  &schemas.TranscriptionInput{File: wav, Filename: "clip.wav"},
			Params: &schemas.TranscriptionParameters{Prompt: prompt},
		})
		if bifrostErr != nil {
			t.Fatalf("Transcription: %s", bifrostErr.GetErrorString())
		}
		select {
		case got := <-requests:
			return got
		default:
			t.Fatal("no request reached the base")
			return sent{}
		}
	}

	t.Run("bearer with default prompt", func(t *testing.T) {
		got := transcribe(key, nil)
		if want := "/v1/projects/project-123/locations/us-central1/publishers/google/models/gemini-3.5-transcribe:generateContent"; got.path != want {
			t.Fatalf("path = %s, want %s", got.path, want)
		}
		if got.query != "" || got.auth != "Bearer test-token" {
			t.Fatalf("query %q auth %q, want no query and the bearer token", got.query, got.auth)
		}
		if got := gjson.Get(got.body, "contents.0.parts.0.text").String(); got != "Generate a transcript of the speech." {
			t.Fatalf("default prompt = %q", got)
		}
		if got := gjson.Get(got.body, "contents.0.parts.1.inlineData.mimeType").String(); got != "audio/wav" {
			t.Fatalf("inline mimeType = %q, want audio/wav from DetectAudioMimeType", got)
		}
		if got, err := base64.StdEncoding.DecodeString(gjson.Get(got.body, "contents.0.parts.1.inlineData.data").String()); err != nil || string(got) != string(wav) {
			t.Fatalf("inlineData does not carry the file bytes: %v %q", err, got)
		}
	})

	t.Run("api key query and custom prompt", func(t *testing.T) {
		prompt := "Summarize the meeting."
		got := transcribe(vertexAudioAPIKey(key), &prompt)
		if got.query != "key=test-api-key" || got.auth != "" {
			t.Fatalf("query %q auth %q, want the api key query and no bearer", got.query, got.auth)
		}
		if got := gjson.Get(got.body, "contents.0.parts.0.text").String(); got != prompt {
			t.Fatalf("prompt = %q, want %q", got, prompt)
		}
	})
}

func TestVertexTranscriptionResponseConversion(t *testing.T) {
	provider, key := vertexAudioTestProvider(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"candidates":[{"content":{"role":"model","parts":[{"text":"hello "},{"text":"world"}]}}],"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":3,"totalTokenCount":15}}`)
	}))
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	response, bifrostErr := provider.Transcription(ctx, key, &schemas.BifrostTranscriptionRequest{
		Provider: schemas.Vertex, Model: "gemini-3.5-transcribe",
		Input: &schemas.TranscriptionInput{File: []byte("RIFF\x00\x00\x00\x00WAVEjunk")},
	})
	if bifrostErr != nil {
		t.Fatalf("Transcription: %s", bifrostErr.GetErrorString())
	}
	if response.Text != "hello world" {
		t.Fatalf("text = %q, want the parts concatenated", response.Text)
	}
	if response.Task == nil || *response.Task != "transcribe" {
		t.Fatalf("task = %v, want transcribe", response.Task)
	}
	if response.Usage == nil || response.Usage.Type != "tokens" ||
		response.Usage.InputTokens == nil || *response.Usage.InputTokens != 12 ||
		response.Usage.OutputTokens == nil || *response.Usage.OutputTokens != 3 ||
		response.Usage.TotalTokens == nil || *response.Usage.TotalTokens != 15 {
		t.Fatalf("usage = %+v, want token-typed 12/3/15", response.Usage)
	}
}

func TestVertexTranscriptionErrorMapping(t *testing.T) {
	provider, key := vertexAudioTestProvider(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":{"code":400,"message":"Audio content is too long.","status":"INVALID_ARGUMENT"}}`)
	}))
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	_, bifrostErr := provider.Transcription(ctx, key, &schemas.BifrostTranscriptionRequest{
		Provider: schemas.Vertex, Model: "gemini-3.5-transcribe",
		Input: &schemas.TranscriptionInput{File: []byte("RIFF\x00\x00\x00\x00WAVEjunk")},
	})
	if bifrostErr == nil || bifrostErr.StatusCode == nil || *bifrostErr.StatusCode != http.StatusBadRequest {
		t.Fatalf("error = %v, want a mapped 400", bifrostErr)
	}
}

func TestVertexTranscriptionStream(t *testing.T) {
	sse := vertexAudioSSE(
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"hello "}]}}]}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"world"}]}}]}`,
		`{"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":3,"totalTokenCount":15}}`,
	)
	provider, key := vertexAudioTestProvider(t, http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, sse)
	}))
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	stream, bifrostErr := provider.TranscriptionStream(ctx, noAudioHooks, nil, key, &schemas.BifrostTranscriptionRequest{
		Provider: schemas.Vertex, Model: "gemini-3.5-transcribe",
		Input: &schemas.TranscriptionInput{File: []byte("RIFF\x00\x00\x00\x00WAVEjunk")},
	})
	if bifrostErr != nil {
		t.Fatalf("TranscriptionStream: %s", bifrostErr.GetErrorString())
	}
	var deltas []string
	var done *schemas.BifrostTranscriptionStreamResponse
	timeout := time.After(5 * time.Second)
	for done == nil {
		select {
		case chunk, ok := <-stream:
			if !ok {
				t.Fatalf("stream closed after %v deltas without a done chunk", deltas)
			}
			if chunk.BifrostError != nil {
				t.Fatalf("stream chunk error: %s", chunk.BifrostError.GetErrorString())
			}
			response := chunk.BifrostTranscriptionStreamResponse
			if response.Type == schemas.TranscriptionStreamResponseTypeDone {
				done = response
			} else if response.Delta != nil {
				deltas = append(deltas, *response.Delta)
			}
		case <-timeout:
			t.Fatal("stream did not finish")
		}
	}
	if len(deltas) != 2 || deltas[0] != "hello " || deltas[1] != "world" {
		t.Fatalf("deltas = %q, want [hello  world]", deltas)
	}
	if done.Text != "hello world" {
		t.Fatalf("done text = %q, want the accumulated transcript", done.Text)
	}
	if done.Usage == nil || done.Usage.Type != "tokens" || done.Usage.TotalTokens == nil || *done.Usage.TotalTokens != 15 {
		t.Fatalf("done usage = %+v, want token-typed usage totalling 15", done.Usage)
	}
}

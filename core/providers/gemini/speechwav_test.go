package gemini

import (
	"context"
	"encoding/base64"
	"testing"

	"github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Vertex gemini-3.8-*-tts answers generateContent with a finished WAV (audio/wav), while
// gemini-2.5-*-tts and every stream chunk carry raw PCM (audio/L16); only PCM gets a header.
func TestToBifrostSpeechResponseWrapsOnlyRawPCM(t *testing.T) {
	pcm := []byte{1, 0, 2, 0, 3, 0, 4, 0}
	wav, err := utils.ConvertPCMToWAV(pcm, utils.DefaultGeminiPCMConfig())
	require.NoError(t, err)

	cases := []struct {
		name string
		mime string
		data []byte
		want []byte
	}{
		{"raw pcm is wrapped", "audio/L16;codec=pcm;rate=24000", pcm, wav},
		{"finished wav passes through", "audio/wav", wav, wav},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			resp := &GenerateContentResponse{Candidates: []*Candidate{{Content: &Content{Parts: []*Part{{
				InlineData: &Blob{MIMEType: tc.mime, Data: base64.StdEncoding.EncodeToString(tc.data)},
			}}}}}}
			ctx := context.WithValue(context.Background(), BifrostContextKeyResponseFormat, "wav")

			got, err := resp.ToBifrostSpeechResponse(ctx)
			require.NoError(t, err)
			assert.Equal(t, tc.want, got.Audio)
		})
	}
}

// Vertex rejects a content without a role ("Please use a valid role: user, model");
// AI Studio defaults it, so only the real Vertex API exposes the gap.
func TestAudioRequestsCarryUserRole(t *testing.T) {
	speech, err := ToGeminiSpeechRequest(nil, &schemas.BifrostSpeechRequest{Model: "gemini-3.8-flash-tts", Input: &schemas.SpeechInput{Input: "hi"}})
	require.NoError(t, err)
	transcription := ToGeminiTranscriptionRequest(&schemas.BifrostTranscriptionRequest{Model: "gemini-3.5-transcribe", Input: &schemas.TranscriptionInput{File: []byte("RIFF____WAVE")}})

	for name, contents := range map[string][]Content{"speech": speech.Contents, "transcription": transcription.Contents} {
		require.Len(t, contents, 1, name)
		assert.Equal(t, RoleUser, contents[0].Role, name)
	}
}

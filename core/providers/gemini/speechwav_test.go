package gemini

import (
	"context"
	"encoding/base64"
	"testing"

	"github.com/maximhq/bifrost/core/providers/utils"
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

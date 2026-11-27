package gemini

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/maximhq/bifrost/core/schemas"
)

// An OpenAI voice name is mapped to its Gemini voice and signalled; a Gemini voice passes unchanged;
// a missing voice gets the default.
func TestToGeminiSpeechRequestMapsOpenAIVoices(t *testing.T) {
	cases := []struct {
		name    string
		params  *schemas.SpeechParameters
		want    string
		adapted bool
	}{
		{"openai alloy", voiceParams("alloy"), "Kore", true},
		{"openai name in any case", voiceParams("ShImMeR"), "Achernar", true},
		{"openai verse", voiceParams("verse"), "Puck", true},
		{"gemini voice unchanged", voiceParams("Zephyr"), "Zephyr", false},
		{"empty voice gets the default", voiceParams(""), "Kore", false},
		{"no voice config gets the default", &schemas.SpeechParameters{}, "Kore", false},
		{"no params gets the default", nil, "Kore", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
			var before *string
			if tc.params != nil && tc.params.VoiceConfig != nil && tc.params.VoiceConfig.Voice != nil {
				v := *tc.params.VoiceConfig.Voice
				before = &v
			}
			request, err := ToGeminiSpeechRequest(ctx, &schemas.BifrostSpeechRequest{
				Model:  "gemini-3.8-flash-tts",
				Input:  &schemas.SpeechInput{Input: "hello"},
				Params: tc.params,
			})
			require.NoError(t, err)
			sc := request.GenerationConfig.SpeechConfig
			require.NotNil(t, sc)
			require.NotNil(t, sc.VoiceConfig)
			require.NotNil(t, sc.VoiceConfig.PrebuiltVoiceConfig)
			assert.Equal(t, tc.want, sc.VoiceConfig.PrebuiltVoiceConfig.VoiceName)
			assert.Equal(t, tc.adapted, schemas.AdaptationsFromContext(ctx).Has(schemas.AdaptationVoice))
			if before != nil {
				assert.Equal(t, *before, *tc.params.VoiceConfig.Voice, "the shared request must not change: a fallback reuses it")
			}
		})
	}
}

// A multi-speaker config is sent as given.
func TestToGeminiSpeechRequestKeepsMultiSpeakerVoices(t *testing.T) {
	request, err := ToGeminiSpeechRequest(nil, &schemas.BifrostSpeechRequest{
		Model: "gemini-3.8-flash-tts",
		Input: &schemas.SpeechInput{Input: "A: hi\nB: hello"},
		Params: &schemas.SpeechParameters{VoiceConfig: &schemas.SpeechVoiceInput{MultiVoiceConfig: []schemas.VoiceConfig{
			{Speaker: "A", Voice: "Puck"}, {Speaker: "B", Voice: "Kore"},
		}}},
	})
	require.NoError(t, err)
	sc := request.GenerationConfig.SpeechConfig
	require.NotNil(t, sc)
	assert.Nil(t, sc.VoiceConfig)
	require.NotNil(t, sc.MultiSpeakerVoiceConfig)
	assert.Len(t, sc.MultiSpeakerVoiceConfig.SpeakerVoiceConfigs, 2)
}

func voiceParams(voice string) *schemas.SpeechParameters {
	return &schemas.SpeechParameters{VoiceConfig: &schemas.SpeechVoiceInput{Voice: &voice}}
}

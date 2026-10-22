package bedrock

import (
	"bytes"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"
	"github.com/stretchr/testify/require"
)

// An open event stream holds about its largest event, not a fixed 1 MiB
// (BIT-2741 B6): the decode buffer starts small, grows to an event that does
// not fit, and keeps that array for the next events instead of reallocating.
func TestStreamPayloadBuffer_GrowsOnDemandAndKeepsTheArray(t *testing.T) {
	var wire bytes.Buffer
	small := []byte(`{"delta":{"text":"Hi"},"contentBlockIndex":0}`)
	large := []byte(`{"delta":{"text":"` + strings.Repeat("x", 100*1024) + `"},"contentBlockIndex":0}`)
	for _, p := range [][]byte{small, large, small} {
		writeEventStreamEvent(t, &wire, "contentBlockDelta", p)
	}

	buf := newStreamPayloadBuffer()
	require.LessOrEqual(t, cap(buf.buf), streamPayloadBufInitialSize)
	decoder := eventstream.NewDecoder()

	m, err := buf.decode(decoder, &wire)
	require.NoError(t, err)
	require.Equal(t, small, m.Payload)
	require.LessOrEqual(t, cap(buf.buf), streamPayloadBufInitialSize, "a small event does not grow the buffer")

	m, err = buf.decode(decoder, &wire)
	require.NoError(t, err)
	require.Equal(t, large, m.Payload)
	grown := cap(buf.buf)
	require.GreaterOrEqual(t, grown, len(large), "the buffer keeps the grown array")

	m, err = buf.decode(decoder, &wire)
	require.NoError(t, err)
	require.Equal(t, small, m.Payload)
	require.Equal(t, grown, cap(buf.buf), "the next event reuses the grown array")
}

// A stream of small events allocates no payload array beyond the initial one.
func TestStreamPayloadBuffer_SmallEventsDoNotAllocatePayload(t *testing.T) {
	var frame bytes.Buffer
	writeEventStreamEvent(t, &frame, "contentBlockDelta", []byte(`{"delta":{"text":"Hi"},"contentBlockIndex":0}`))
	raw := frame.Bytes()
	buf := newStreamPayloadBuffer()
	decoder := eventstream.NewDecoder()
	r := bytes.NewReader(raw)
	base := testing.AllocsPerRun(50, func() {
		r.Reset(raw)
		if _, err := decoder.Decode(r, make([]byte, 0, streamPayloadBufInitialSize)); err != nil {
			t.Fatal(err)
		}
	})
	got := testing.AllocsPerRun(50, func() {
		r.Reset(raw)
		if _, err := buf.decode(decoder, r); err != nil {
			t.Fatal(err)
		}
	})
	require.Less(t, got, base, "reusing the stream's buffer saves the payload array")
}

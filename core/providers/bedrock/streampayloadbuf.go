package bedrock

import (
	"io"

	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"
)

// streamPayloadBufInitialSize is the initial capacity of an event-stream
// decode buffer: room for every ordinary converse event (text and tool deltas,
// metadata) without growing.
const streamPayloadBufInitialSize = 4 * 1024

// streamPayloadBuffer is the payload scratch buffer of one event-stream decode
// loop. It starts small and keeps the largest backing array an event needed,
// so an open stream holds about its largest event instead of a fixed 1 MiB
// allocated up front (1 MiB x 1000 open streams was 1 GiB of heap).
//
// As with a fixed buffer, a decoded Message.Payload aliases the buffer and is
// valid only until the next decode.
type streamPayloadBuffer struct {
	buf []byte
}

func newStreamPayloadBuffer() streamPayloadBuffer {
	return streamPayloadBuffer{buf: make([]byte, 0, streamPayloadBufInitialSize)}
}

// decode reads one message. An event larger than the buffer is decoded into a
// grown array (the decoder appends into the buffer), which the buffer then
// keeps for the next events.
func (b *streamPayloadBuffer) decode(decoder *eventstream.Decoder, r io.Reader) (eventstream.Message, error) {
	message, err := decoder.Decode(r, b.buf)
	if cap(message.Payload) > cap(b.buf) {
		b.buf = message.Payload[:0]
	}
	return message, err
}

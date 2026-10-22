package utils

import (
	"context"
	"strings"
	"testing"

	schemas "github.com/maximhq/bifrost/core/schemas"
)

// Sizing a startup event for the preamble budget measures its encoded length
// without keeping an encoded copy: response.created and response.in_progress
// carry the whole response object, and both the producer and the retry
// reader size each of them, so a fresh marshal per check cost a stream about
// 10 KB of garbage per preamble pair (BIT-2741 R3). The measured size is the
// MarshalSorted length the budget was defined in.
func TestFitStreamPreamble_SizesWithoutAllocatingTheEncoding(t *testing.T) {
	instructions := strings.Repeat("You are a helpful assistant. ", 200)
	chunk := &schemas.BifrostStreamChunk{BifrostResponsesStreamResponse: &schemas.BifrostResponsesStreamResponse{
		Type: schemas.ResponsesStreamResponseTypeCreated,
		Response: &schemas.BifrostResponsesResponse{
			ID:           schemas.Ptr("resp_1"),
			Model:        "gpt-5",
			Instructions: &schemas.ResponsesResponseInstructions{ResponsesResponseInstructionsStr: &instructions},
		},
	}}
	encoded, err := MarshalSorted(chunk)
	if err != nil {
		t.Fatal(err)
	}
	size, ok := fitStreamPreamble(0, 0, chunk, maxStreamPreambleBytes)
	if !ok || size != len(encoded) {
		t.Fatalf("fitStreamPreamble = %d, %v; want %d, true", size, ok, len(encoded))
	}
	if _, ok := fitStreamPreamble(0, maxStreamPreambleBytes-len(encoded), chunk, maxStreamPreambleBytes); ok {
		t.Fatal("a chunk that does not fit the remaining budget must be refused")
	}

	if raceEnabled {
		return // the race detector drops sync.Pool items at random; the allocation check needs a stable pool
	}
	marshal := testing.AllocsPerRun(20, func() { _, _ = MarshalSorted(chunk) })
	fit := testing.AllocsPerRun(20, func() { _, _ = fitStreamPreamble(0, 0, chunk, maxStreamPreambleBytes) })
	if fit >= marshal {
		t.Fatalf("fitStreamPreamble allocates %.0f per check, MarshalSorted %.0f: the size check must not keep an encoded copy", fit, marshal)
	}
}

// Each startup event is sized ONCE per attempt (BIT-2741 R3): the producer
// (StreamStartupState.ObserveResult) sizes it and records the size against
// the event, and the retry reader (CheckStreamPreambleForError) reuses that
// size for the same event instead of encoding it a second time. A chunk the
// producer did not size (or a different object) is still sized by the reader.
func TestStreamPreamble_EachStartupEventSizedOnce(t *testing.T) {
	var sizings int
	orig := preambleEncodedSize
	preambleEncodedSize = func(v any) (int, error) { sizings++; return orig(v) }
	t.Cleanup(func() { preambleEncodedSize = orig })

	isPreamble := func(c *schemas.BifrostStreamChunk) bool {
		r := c.BifrostResponsesStreamResponse
		return r != nil && (r.Type == schemas.ResponsesStreamResponseTypeCreated || r.Type == schemas.ResponsesStreamResponseTypeInProgress)
	}
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	startup := NewStreamStartupState(isPreamble, false)
	startup.Attach(ctx)

	stream := make(chan *schemas.BifrostStreamChunk, 4)
	for _, typ := range []schemas.ResponsesStreamResponseType{schemas.ResponsesStreamResponseTypeCreated, schemas.ResponsesStreamResponseTypeInProgress} {
		resp := &schemas.BifrostResponse{ResponsesStreamResponse: &schemas.BifrostResponsesStreamResponse{
			Type: typ, Response: &schemas.BifrostResponsesResponse{ID: schemas.Ptr("resp_1"), Model: "gpt-5"},
		}}
		startup.ObserveResult(ctx, resp, nil)            // the producer sizes it
		stream <- BuildClientStreamChunk(ctx, resp, nil) // and sends the same event
	}
	stream <- &schemas.BifrostStreamChunk{BifrostResponsesStreamResponse: &schemas.BifrostResponsesStreamResponse{
		Type: schemas.ResponsesStreamResponseTypeOutputTextDelta, Delta: schemas.Ptr("hi"),
	}}
	close(stream)

	out, done, bifrostErr := CheckStreamPreambleForError(ctx, "req-1", stream, isPreamble)
	if bifrostErr != nil {
		t.Fatalf("unexpected startup error: %+v", bifrostErr)
	}
	var n int
	for range out {
		n++
	}
	<-done
	if n != 3 {
		t.Fatalf("replayed %d chunks, want 3", n)
	}
	if sizings != 2 {
		t.Fatalf("startup events sized %d times, want 2 (once each)", sizings)
	}
}

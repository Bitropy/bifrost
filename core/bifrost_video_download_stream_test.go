package bifrost

import (
	"context"
	"strings"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

// Validation refusals fire before the pipeline runs, so a zero-value Bifrost is enough to
// exercise them; each names the streamed-download request type.
func TestVideoDownloadStreamRequestValidation(t *testing.T) {
	bifrost := &Bifrost{}
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)

	tests := []struct {
		name string
		req  *schemas.BifrostVideoDownloadStreamRequest
		want string
	}{
		{"nil request", nil, "video download request is nil"},
		{"empty provider", &schemas.BifrostVideoDownloadStreamRequest{ID: "op"}, "provider is required"},
		{"empty id", &schemas.BifrostVideoDownloadStreamRequest{Provider: schemas.Vertex}, "video_id is required"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader, contentType, bifrostErr := bifrost.VideoDownloadStreamRequest(ctx, tt.req)
			if bifrostErr == nil {
				t.Fatal("validation refusal missing")
			}
			if reader != nil || contentType != "" {
				t.Fatalf("refusal returned reader=%v contentType=%q", reader, contentType)
			}
			if msg := bifrostErr.GetErrorString(); !strings.Contains(msg, tt.want) {
				t.Fatalf("error = %q, want it to contain %q", msg, tt.want)
			}
			if got := bifrostErr.ExtraFields.RequestType; got != schemas.VideoDownloadStreamRequest {
				t.Fatalf("request type = %q, want %q", got, schemas.VideoDownloadStreamRequest)
			}
		})
	}
}

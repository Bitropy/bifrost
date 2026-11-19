package openai

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"unsafe"

	"github.com/tidwall/gjson"
)

// openAIResponsesRequestKnownFields are the top-level keys OpenAIResponsesRequest decodes, read
// from its JSON tags (the embedded ResponsesParameters included), plus Bifrost's own extra_params
// wrapper. Lowercased: the decoder matches a key to a field case-insensitively.
var openAIResponsesRequestKnownFields = func() map[string]bool {
	known := map[string]bool{"extra_params": true}
	addJSONFieldNames(reflect.TypeOf(OpenAIResponsesRequest{}), known)
	folded := make(map[string]bool, len(known))
	for name := range known {
		folded[strings.ToLower(name)] = true
	}
	return folded
}()

// isKnownOpenAIResponsesRequestField reports whether the decoder takes name, folded to ASCII lower
// case in a stack buffer so the lookup does not allocate.
func isKnownOpenAIResponsesRequestField(name string) bool {
	var buf [64]byte
	if len(name) > len(buf) {
		return openAIResponsesRequestKnownFields[strings.ToLower(name)]
	}
	for i := 0; i < len(name); i++ {
		c := name[i]
		if 'A' <= c && c <= 'Z' {
			c += 'a' - 'A'
		}
		buf[i] = c
	}
	return openAIResponsesRequestKnownFields[string(buf[:len(name)])]
}

// OpenAIResponsesRequestUnknownFields returns the top-level fields of a Responses body that
// OpenAIResponsesRequest does not decode, each compacted as a json.RawMessage, or nil when there
// are none: the Responses counterpart of OpenAIChatRequestUnknownFields. Decoding is unchanged; an
// embedder that forwards unknown keys puts the result in Params.ExtraParams. A body with only known
// fields costs a scan and no allocation.
func OpenAIResponsesRequestUnknownFields(data []byte) map[string]interface{} {
	if len(data) == 0 {
		return nil
	}
	var unknown map[string]interface{}
	gjson.Parse(unsafe.String(unsafe.SliceData(data), len(data))).ForEach(func(key, value gjson.Result) bool {
		name := key.String()
		if isKnownOpenAIResponsesRequestField(name) {
			return true
		}
		if unknown == nil {
			unknown = make(map[string]interface{})
		}
		var compact bytes.Buffer
		if err := json.Compact(&compact, []byte(value.Raw)); err == nil {
			unknown[strings.Clone(name)] = json.RawMessage(compact.Bytes())
		} else {
			unknown[strings.Clone(name)] = json.RawMessage(value.Raw)
		}
		return true
	})
	return unknown
}

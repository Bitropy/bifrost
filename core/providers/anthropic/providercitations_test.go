package anthropic

import (
	"fmt"
	"testing"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/core/schemas"
)

const (
	citeChar    = `{"type":"char_location","cited_text":"alpha","document_index":0,"document_title":"Doc","start_char_index":0,"end_char_index":5}`
	citePage    = `{"type":"page_location","cited_text":"beta","document_index":1,"start_page_number":2,"end_page_number":3}`
	citeFuture  = `{"type":"future_location","cited_text":"gamma","future_member":{"x":[1]}}`
	citeWeb     = `{"type":"web_search_result_location","url":"https://a.example/x","title":"A","encrypted_index":"E","cited_text":"web"}`
	citedFirst  = "Alpha and beta."
	citedSecond = " Web says so."
)

// wantProviderCitation is one expected provider_citation annotation.
type wantProviderCitation struct {
	start, end int
	citation   string
}

func checkProviderCitations(t *testing.T, anns []schemas.ChatAssistantMessageAnnotation, wantURL int, want []wantProviderCitation) {
	t.Helper()
	// Content order; within a block url_citations come first.
	wantOrder := []string{"provider_citation", "provider_citation", "url_citation", "provider_citation"}
	var urls, providers []schemas.ChatAssistantMessageAnnotation
	var order []string
	for _, a := range anns {
		order = append(order, a.Type)
		switch a.Type {
		case "url_citation":
			urls = append(urls, a)
		case schemas.ChatAnnotationTypeProviderCitation:
			providers = append(providers, a)
		}
	}
	if fmt.Sprint(order) != fmt.Sprint(wantOrder) {
		t.Errorf("annotation order = %v, want %v", order, wantOrder)
	}
	if len(urls) != wantURL {
		t.Errorf("url_citation annotations = %d, want %d", len(urls), wantURL)
	}
	if len(providers) != len(want) {
		t.Fatalf("provider_citation annotations = %+v, want %d", providers, len(want))
	}
	for i, w := range want {
		a := providers[i]
		if a.Provider != schemas.ProviderBlockAnthropic || a.StartIndex == nil || a.EndIndex == nil || *a.StartIndex != w.start || *a.EndIndex != w.end {
			t.Errorf("annotation %d = {provider %q, span %v..%v}, want {anthropic, %d..%d}", i, a.Provider, a.StartIndex, a.EndIndex, w.start, w.end)
		}
		if got := compactJSON(t, a.Citation); got != w.citation {
			t.Errorf("annotation %d citation = %s, want %s", i, got, w.citation)
		}
	}
}

var wantDocCitations = []wantProviderCitation{
	{0, len(citedFirst), citeChar},
	{0, len(citedFirst), citePage},
	{len(citedFirst), len(citedFirst) + len(citedSecond), citeFuture},
}

func TestToBifrostChatResponse_ProviderCitations(t *testing.T) {
	body := unaryWith(
		`{"type":"text","text":"`+citedFirst+`","citations":[`+citeChar+`,`+citePage+`]}`,
		`{"type":"text","text":"`+citedSecond+`","citations":[`+citeWeb+`,`+citeFuture+`]}`,
	)
	am := convertUnary(t, body, true)
	if am == nil {
		t.Fatal("no assistant message")
	}
	checkProviderCitations(t, am.Annotations, 1, wantDocCitations)
}

func TestChatStream_ProviderCitationsAtBlockStop(t *testing.T) {
	out := feedChatStream(t, []string{
		startEvent("0", `{"type":"text","text":"","citations":[`+citeChar+`]}`),
		`{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":` + citePage + `}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"` + citedFirst + `"}}`,
		stopEvent("0"),
		startEvent("1", `{"type":"text","text":""}`),
		`{"type":"content_block_delta","index":1,"delta":{"type":"citations_delta","citation":` + citeWeb + `}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"citations_delta","citation":` + citeFuture + `}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"` + citedSecond + `"}}`,
		stopEvent("1"),
	})
	var anns []schemas.ChatAssistantMessageAnnotation
	for i, c := range out.chunks {
		if a := c.Choices[0].Delta.Annotations; len(a) > 0 {
			if ev := out.atEvent[i]; ev != 3 && ev != 8 {
				t.Errorf("annotations emitted on event %d, want only the blocks' stops", ev)
			}
			anns = append(anns, a...)
		}
	}
	checkProviderCitations(t, anns, 1, wantDocCitations)
}

func TestProviderCitationEncoding(t *testing.T) {
	start, end := 0, 5
	got, err := sonic.Marshal(schemas.ChatAssistantMessageAnnotation{
		Type: schemas.ChatAnnotationTypeProviderCitation, Provider: schemas.ProviderBlockAnthropic,
		StartIndex: &start, EndIndex: &end, Citation: []byte(citeChar),
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := `{"type":"provider_citation","provider":"anthropic","start_index":0,"end_index":5,"citation":` + citeChar + `}`; string(got) != want {
		t.Errorf("provider_citation = %s\nwant %s", got, want)
	}
	// url_citation bytes are what they were before the provider members existed.
	got, err = sonic.Marshal(schemas.ChatAssistantMessageAnnotation{
		Type:        "url_citation",
		URLCitation: schemas.ChatAssistantMessageAnnotationCitation{StartIndex: 1, EndIndex: 4, Title: "A", URL: schemas.Ptr("https://a"), Text: schemas.Ptr("x")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := `{"type":"url_citation","url_citation":{"start_index":1,"end_index":4,"title":"A","url":"https://a","text":"x"}}`; string(got) != want {
		t.Errorf("url_citation = %s\nwant %s", got, want)
	}
}

// An answer whose only annotations are url_citations encodes with the base pin's
// allocations.
func TestChatAnnotationEncode_URLCitationOnly_AllocBudget(t *testing.T) {
	skipAllocUnderRace(t)
	const urlCitationEncodeAllocs = 19
	var resp AnthropicMessageResponse
	if err := sonic.Unmarshal([]byte(noProviderBlockUnary), &resp); err != nil {
		t.Fatal(err)
	}
	out := resp.ToBifrostChatResponse(nil)
	got := testing.AllocsPerRun(200, func() { _, _ = sonic.Marshal(out) })
	t.Logf("url_citation-only answer encodes in %v allocs", got)
	if got > urlCitationEncodeAllocs {
		t.Fatalf("encoding allocates %v, budget %d", got, urlCitationEncodeAllocs)
	}
}

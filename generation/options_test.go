package generation_test

import (
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/JWset/gofish/generation"
)

func TestStreamStopsAcrossChunks(t *testing.T) {
	var streamed strings.Builder
	c, err := generation.Resolve([]generation.Option{generation.WithStop("<end>"), generation.WithStream(func(s string) error { streamed.WriteString(s); return nil })})
	if err != nil {
		t.Fatal(err)
	}
	collector := generation.NewCollector(c)
	for _, chunk := range []string{"hello ", "🐟", "<e", "nd>trailing"} {
		if _, err = collector.Feed(chunk); err != nil {
			t.Fatal(err)
		}
	}
	if err = collector.Finish(); err != nil {
		t.Fatal(err)
	}
	if collector.Text() != "hello 🐟" || streamed.String() != collector.Text() {
		t.Fatalf("stop leaked into output: %q, %q", collector.Text(), streamed.String())
	}
}
func TestPartialUTF8AndCallbackError(t *testing.T) {
	c, _ := generation.Resolve(nil)
	collector := generation.NewCollector(c)
	fish := []byte("🐟")
	for _, b := range fish {
		if _, err := collector.Feed(string([]byte{b})); err != nil {
			t.Fatal(err)
		}
	}
	if err := collector.Finish(); err != nil {
		t.Fatal(err)
	}
	if collector.Text() != "🐟" {
		t.Fatal(collector.Text())
	}
	want := errors.New("stop streaming")
	c, _ = generation.Resolve([]generation.Option{generation.WithStream(func(string) error { return want })})
	if _, err := generation.NewCollector(c).Feed("x"); !errors.Is(err, want) {
		t.Fatal(err)
	}
	for _, options := range [][]generation.Option{{generation.WithMaxTokens(-1)}, {generation.WithTemperature(math.NaN())}, {generation.WithStop("")}, {nil}} {
		if _, err := generation.Resolve(options); err == nil {
			t.Fatal("accepted invalid options")
		}
	}
}

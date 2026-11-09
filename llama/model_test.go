package llama

import (
	"context"
	"math"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/JWset/gofish/generation"
	"github.com/JWset/gofish/training"
)

func small(t *testing.T) *Model {
	t.Helper()
	m, err := NewByteModel(Config{ContextSize: 8, Width: 8, Heads: 2, KVHeads: 1, Layers: 2, FeedForward: 16, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}
	return m
}
func TestKVCacheMatchesFullAutogradGraph(t *testing.T) {
	m := small(t)
	ids := []int{256, 'h', 'e', 'l', 'l', 'o'}
	weights, _, err := m.parameters(false)
	if err != nil {
		t.Fatal(err)
	}
	full := m.forward(ids, 1, weights).Data()
	session := m.NewSession()
	for position, id := range ids {
		logits, err := session.Step(context.Background(), id)
		if err != nil {
			t.Fatal(err)
		}
		for i, got := range logits {
			want := full[position*257+i]
			if math.Abs(got-want) > 2e-6 {
				t.Fatalf("position=%d token=%d cached=%g full=%g", position, i, got, want)
			}
		}
	}
	for i := 0; i < 20; i++ {
		if _, err = session.Step(context.Background(), 'a'); err != nil {
			t.Fatal(err)
		}
	}
	session.Reset()
	again, err := session.Step(context.Background(), 256)
	if err != nil || math.Abs(again[3]-full[3]) > 2e-6 {
		t.Fatal("cache reset", err)
	}
}
func TestTrainExportReload(t *testing.T) {
	m := small(t)
	text := strings.Repeat("hello fish.\n", 8)
	ctx := context.Background()
	before, err := m.Loss(ctx, text)
	if err != nil {
		t.Fatal(err)
	}
	old := m.NewSession()
	if _, err = m.Train(ctx, text, training.Options{Steps: 160, BatchSize: 2, LearningRate: 0.01, Seed: 42}); err != nil {
		t.Fatal(err)
	}
	after, err := m.Loss(ctx, text)
	if err != nil || after > before*0.25 {
		t.Fatal("did not converge", before, after, err)
	}
	if _, err = old.Step(ctx, 256); err == nil {
		t.Fatal("accepted stale cache")
	}
	path := filepath.Join(t.TempDir(), "model.gguf")
	if err = m.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	want, err := m.Generate(ctx, "hello", generation.WithTemperature(0), generation.WithMaxTokens(20))
	if err != nil {
		t.Fatal(err)
	}
	got, err := loaded.Generate(ctx, "hello", generation.WithTemperature(0), generation.WithMaxTokens(20))
	if err != nil || got != want {
		t.Fatal("roundtrip", got, want, err)
	}
	if _, err = loaded.Train(ctx, text, training.Options{Steps: 1, BatchSize: 1}); err != nil {
		t.Fatal(err)
	}
	for _, prompt := range []string{"你好 🐟", "", "café"} {
		out, err := loaded.Generate(ctx, prompt, generation.WithMaxTokens(5))
		if err != nil || !utf8.ValidString(out) {
			t.Fatal(out, err)
		}
	}
	ctx, cancel := context.WithCancel(ctx)
	cancel()
	if _, err = loaded.Generate(ctx, "hello"); err == nil {
		t.Fatal("ignored cancellation")
	}
}
func TestBundledGGUF(t *testing.T) {
	m, err := Pretrained()
	if err != nil {
		t.Fatal(err)
	}
	if m.ParameterCount() != 25760 {
		t.Fatal(m.ParameterCount())
	}
	text, err := m.Generate(context.Background(), "the little fish", generation.WithTemperature(0), generation.WithMaxTokens(80))
	if err != nil || !strings.Contains(text, "sea") {
		t.Fatal(text, err)
	}
}
func TestSentencePieceMergingAndByteFallback(t *testing.T) {
	v := &vocabulary{model: "llama", bos: 256, eos: 257, addBOS: true, spacePrefix: true, lookup: map[string]int{}}
	for b := 0; b < 256; b++ {
		v.bytes[b] = b
		v.pieces = append(v.pieces, "byte")
		v.types = append(v.types, 6)
		v.scores = append(v.scores, 0)
	}
	for _, s := range []string{"<s>", "</s>", "▁", "h", "i", "hi", "▁hi"} {
		v.lookup[s] = len(v.pieces)
		v.pieces = append(v.pieces, s)
		v.types = append(v.types, 1)
		v.scores = append(v.scores, float32(len(s)))
	}
	v.types[256] = 3
	v.types[257] = 3
	ids, err := v.encode("hi 🐟")
	if err != nil {
		t.Fatal(err)
	}
	if ids[0] != 256 || ids[1] != 262 {
		t.Fatal("SPM merges", ids)
	}
	// Metaspace becomes a normal token; unknown Unicode characters use UTF-8 bytes.
	if len(ids) != 7 || ids[2] != 258 || ids[3] != 0xf0 || ids[4] != 0x9f || ids[5] != 0x90 || ids[6] != 0x9f {
		t.Fatal("fallback", ids)
	}
}

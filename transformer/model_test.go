package transformer_test

import (
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"

	"github.com/JWset/gofish/generation"
	"github.com/JWset/gofish/tokenizer"
	"github.com/JWset/gofish/training"
	"github.com/JWset/gofish/transformer"
)

func TestBundledTransformerIsTrainedAndResumable(t *testing.T) {
	m, err := transformer.Pretrained()
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join("corpora", "tiny-transformer.txt"))
	if err != nil {
		t.Fatal(err)
	}
	loss, err := m.Loss(context.Background(), string(data))
	if err != nil {
		t.Fatal(err)
	}
	if loss > 0.6 || m.Updates() != 6000 || m.ParameterCount() != 34881 {
		t.Fatalf("unexpected pretrained model: loss=%g updates=%d parameters=%d", loss, m.Updates(), m.ParameterCount())
	}
	text, err := m.Generate(context.Background(), "the little fish", generation.WithTemperature(0), generation.WithMaxTokens(60))
	if err != nil || !utf8.ValidString(text) || !strings.Contains(text, "sea") {
		t.Fatalf("unexpected pretrained continuation %q: %v", text, err)
	}
	if _, err = m.Train(context.Background(), string(data), training.Options{Steps: 1, BatchSize: 1, Seed: 42, Resume: true}); err != nil {
		t.Fatalf("pretrained checkpoint lacks trainable optimizer state: %v", err)
	}
}

func config() transformer.Config {
	return transformer.Config{ContextSize: 8, Width: 8, Heads: 2, Layers: 1, FeedForward: 16, Seed: 42}
}
func newModel(t *testing.T) *transformer.Model {
	t.Helper()
	m, err := transformer.New(config())
	if err != nil {
		t.Fatal(err)
	}
	return m
}
func equalWeights(t *testing.T, a, b *transformer.Model) {
	t.Helper()
	for i, p := range a.Parameters() {
		if !slices.Equal(p.Data(), b.Parameters()[i].Data()) {
			t.Fatalf("parameter %d differs", i)
		}
	}
}

func TestTrainingConverges(t *testing.T) {
	m := newModel(t)
	text := strings.Repeat("the fish swims.\n", 8)
	ctx := context.Background()
	before, err := m.Loss(ctx, text)
	if err != nil {
		t.Fatal(err)
	}
	history, err := m.Train(ctx, text, training.Options{Steps: 160, BatchSize: 2, LearningRate: 0.01, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}
	after, err := m.Loss(ctx, text)
	if err != nil {
		t.Fatal(err)
	}
	if history.Steps != 160 || after > before*0.2 {
		t.Fatalf("transformer did not converge: %g -> %g", before, after)
	}
	text, err = m.Generate(ctx, "the fish", generation.WithMaxTokens(20), generation.WithTemperature(0))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(text, " swims.\n") {
		t.Fatalf("learned continuation: %q", text)
	}
}
func TestCheckpointResumesExactly(t *testing.T) {
	ctx := context.Background()
	corpus := "abc abc abc abc abc abc abc abc abc\n"
	a := newModel(t)
	opts := training.Options{Steps: 8, BatchSize: 2, LearningRate: 0.01, WeightDecay: 0.02, Seed: 17}
	if _, err := a.Train(ctx, corpus, opts); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "resumable.gofish")
	if err := a.Save(path); err != nil {
		t.Fatal(err)
	}
	b, err := transformer.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	equalWeights(t, a, b)
	if b.Updates() != 8 {
		t.Fatal("lost update count")
	}
	left, err := a.Generate(ctx, "abc", generation.WithMaxTokens(25), generation.WithSeed(42))
	if err != nil {
		t.Fatal(err)
	}
	right, err := b.Generate(ctx, "abc", generation.WithMaxTokens(25), generation.WithSeed(42))
	if err != nil || left != right {
		t.Fatal("generation changed after reload", err)
	}
	opts.Resume = true
	opts.Steps = 5
	if _, err = a.Train(ctx, corpus, opts); err != nil {
		t.Fatal(err)
	}
	if _, err = b.Train(ctx, corpus, opts); err != nil {
		t.Fatal(err)
	}
	equalWeights(t, a, b)
	c := newModel(t)
	opts.Resume = false
	opts.Steps = 13
	if _, err = c.Train(ctx, corpus, opts); err != nil {
		t.Fatal(err)
	}
	equalWeights(t, a, c)
}
func TestUntrainedCheckpointUnicodeAndCancellation(t *testing.T) {
	m := newModel(t)
	path := filepath.Join(t.TempDir(), "untrained.gofish")
	if err := m.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := transformer.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	equalWeights(t, m, loaded)
	for _, n := range []int{1, 2, 3, 4, 9} {
		var stream strings.Builder
		text, err := m.Generate(context.Background(), "你好 🐟 café", generation.WithMaxTokens(n), generation.WithSeed(int64(n)), generation.WithStream(func(s string) error {
			if !utf8.ValidString(s) {
				t.Errorf("invalid UTF-8 chunk %q", s)
			}
			stream.WriteString(s)
			return nil
		}))
		if err != nil || !utf8.ValidString(text) || len(text) > n || text != stream.String() {
			t.Fatalf("invalid byte generation %q: %v", text, err)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := m.Generate(ctx, "hello"); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := m.Train(ctx, "hello", training.Options{}); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := m.Generate(context.Background(), string([]byte{0xff})); err == nil {
		t.Fatal("accepted invalid UTF-8 prompt")
	}
	if _, err := m.Train(context.Background(), "hi", training.Options{Resume: true}); err == nil {
		t.Fatal("resumed without optimizer state")
	}
	for _, data := range [][]byte{nil, []byte("invalid"), {0x1f, 0x8b}} {
		if _, err := transformer.LoadBytes(data); err == nil {
			t.Fatal("accepted corrupt checkpoint")
		}
	}
}
func TestFullTransformerCausalityAndParameterCount(t *testing.T) {
	m := newModel(t)
	ids := []int{tokenizer.BOS, 'a', 'b', 'c'}
	before := m.Forward(ids, 1).Data()
	ids[3] = 'x'
	after := m.Forward(ids, 1).Data()
	if !slices.Equal(before[:3*tokenizer.VocabularySize], after[:3*tokenizer.VocabularySize]) {
		t.Fatal("future token leaked through decoder")
	}
	n := 0
	for _, p := range m.Parameters() {
		n += p.Numel()
	}
	if n != m.ParameterCount() {
		t.Fatal(n, m.ParameterCount())
	}
	for _, c := range []transformer.Config{{Width: 7, Heads: 2}, {Layers: -1}, {ContextSize: 257}, {Width: 512, Layers: 8}} {
		if _, err := transformer.New(c); err == nil {
			t.Fatal("accepted invalid or excessive configuration", c)
		}
	}
}
func TestConcurrentOperations(t *testing.T) {
	m := newModel(t)
	var group sync.WaitGroup
	for i := 0; i < 3; i++ {
		group.Add(1)
		go func() {
			defer group.Done()
			if _, err := m.Generate(context.Background(), "🐟", generation.WithMaxTokens(3)); err != nil {
				t.Error(err)
			}
		}()
	}
	group.Add(1)
	go func() {
		defer group.Done()
		if _, err := m.Train(context.Background(), "the fish swims", training.Options{Steps: 2, BatchSize: 1}); err != nil {
			t.Error(err)
		}
	}()
	group.Wait()
}
func TestInvalidTrainingOptions(t *testing.T) {
	m := newModel(t)
	for _, opts := range []training.Options{{Steps: -1}, {BatchSize: -1}, {LearningRate: math.NaN()}, {WeightDecay: -1}, {ClipNorm: math.Inf(1)}} {
		if _, err := m.Train(context.Background(), "hello", opts); err == nil {
			t.Fatal("accepted invalid options", opts)
		}
	}
}

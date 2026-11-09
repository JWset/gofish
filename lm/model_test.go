package lm_test

import (
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/JWset/gofish/generation"
	"github.com/JWset/gofish/lm"
)

func TestTrainSaveLoadAndFineTune(t *testing.T) {
	ctx := context.Background()
	text := "hello fish. hello bird. hello fish. hello bird.\n"
	m, err := lm.New(text, lm.Config{ContextSize: 4, EmbeddingSize: 4, HiddenSize: 16, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}
	before, err := m.Loss(ctx, text)
	if err != nil {
		t.Fatal(err)
	}
	history, err := m.Train(ctx, text, lm.TrainOptions{Steps: 150, BatchSize: 16, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}
	after, err := m.Loss(ctx, text)
	if err != nil {
		t.Fatal(err)
	}
	if len(history.Losses) != 150 || after > before*0.3 {
		t.Fatalf("training did not converge: %g -> %g", before, after)
	}
	path := filepath.Join(t.TempDir(), "model.gofish")
	if err = m.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := lm.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	opts := []generation.Option{generation.WithSeed(42), generation.WithMaxTokens(50)}
	want, err := m.Generate(ctx, "hello", opts...)
	if err != nil {
		t.Fatal(err)
	}
	got, err := loaded.Generate(ctx, "hello", opts...)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("round trip changed generation: %q vs %q", got, want)
	}
	if _, err = loaded.Train(ctx, text, lm.TrainOptions{Steps: 2}); err != nil {
		t.Fatalf("fine-tuning loaded model: %v", err)
	}
	ctx, cancel := context.WithCancel(ctx)
	cancel()
	if _, err = m.Generate(ctx, "hello"); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err = m.Train(ctx, text, lm.TrainOptions{Steps: 2}); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err = m.Generate(context.Background(), "🙂"); err == nil {
		t.Fatal("accepted unknown character")
	}
	if _, err = m.Train(context.Background(), text, lm.TrainOptions{LearningRate: math.NaN()}); err == nil {
		t.Fatal("accepted invalid training options")
	}
}

func TestPretrainedModels(t *testing.T) {
	for _, name := range []string{"tiny-stories", "tiny-dialogue"} {
		t.Run(name, func(t *testing.T) {
			m, err := lm.Pretrained(name)
			if err != nil {
				t.Fatal(err)
			}
			corpus, err := os.ReadFile(filepath.Join("corpora", name+".txt"))
			if err != nil {
				t.Fatal(err)
			}
			loss, err := m.Loss(context.Background(), string(corpus))
			if err != nil {
				t.Fatal(err)
			}
			if loss > 0.15 {
				t.Fatalf("premade model has not learned its corpus: loss=%g", loss)
			}
			prompt := "the sun was"
			expected := " warm and the river was quiet."
			if name == "tiny-dialogue" {
				prompt = "user: hello\nfish:"
				expected = " hello! how are you today?"
			}
			generated, err := m.Generate(context.Background(), prompt, generation.WithMaxTokens(60), generation.WithTemperature(0))
			if err != nil {
				t.Fatal(err)
			}
			if !strings.HasPrefix(generated, expected) {
				t.Fatalf("premade continuation %q does not start with %q", generated, expected)
			}
			if _, err = m.Train(context.Background(), string(corpus), lm.TrainOptions{Steps: 1}); err != nil {
				t.Fatalf("premade model not trainable: %v", err)
			}
		})
	}
}
func TestUnicodeAndInvalidCheckpoints(t *testing.T) {
	m, err := lm.New("🐟你好🐟你好", lm.Config{ContextSize: 2, EmbeddingSize: 2, HiddenSize: 4})
	if err != nil {
		t.Fatal(err)
	}
	text, err := m.Generate(context.Background(), "你好", generation.WithMaxTokens(4))
	if err != nil {
		t.Fatal(err)
	}
	if len([]rune(text)) != 4 {
		t.Fatal(text)
	}
	for _, data := range [][]byte{nil, []byte("not a checkpoint"), {0x1f, 0x8b}} {
		if _, err = lm.LoadBytes(data); err == nil {
			t.Fatal("accepted invalid checkpoint")
		}
	}
	if _, err = lm.New("a\x00b", lm.Config{}); err == nil {
		t.Fatal("accepted reserved character")
	}
	if _, err = lm.New("abc", lm.Config{HiddenSize: -1}); err == nil {
		t.Fatal("accepted bad dimensions")
	}
}

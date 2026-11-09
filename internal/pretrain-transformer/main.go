// Command pretrain-transformer regenerates the bundled native transformer.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/JWset/gofish/generation"
	"github.com/JWset/gofish/training"
	"github.com/JWset/gofish/transformer"
)

func main() {
	steps := flag.Int("steps", 6000, "AdamW updates")
	resume := flag.String("resume", "", "resume from an existing transformer checkpoint")
	out := flag.String("out", "transformer/assets/tiny-transformer.gofish", "checkpoint destination")
	flag.Parse()
	corpus, err := os.ReadFile("transformer/corpora/tiny-transformer.txt")
	if err != nil {
		log.Fatal(err)
	}
	m, err := transformer.New(transformer.Config{ContextSize: 32, Width: 32, Heads: 4, Layers: 2, FeedForward: 64, Seed: 42})
	if *resume != "" {
		m, err = transformer.Load(*resume)
	}
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	initial, err := m.Loss(ctx, string(corpus))
	if err != nil {
		log.Fatal(err)
	}
	start := time.Now()
	fmt.Printf("parameters=%d initial_loss=%.4f\n", m.ParameterCount(), initial)
	_, err = m.Train(ctx, string(corpus), training.Options{Steps: *steps, BatchSize: 4, LearningRate: 0.003, WeightDecay: 0.01, Seed: 42, Resume: *resume != "", OnStep: func(s training.Step) {
		if s.Number%200 == 0 {
			fmt.Printf("step=%d loss=%.4f elapsed=%s\n", s.Number, s.Loss, time.Since(start).Round(time.Second))
		}
	}})
	if err != nil {
		log.Fatal(err)
	}
	final, err := m.Loss(ctx, string(corpus))
	if err != nil {
		log.Fatal(err)
	}
	if err = m.Save(*out); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("updates=%d corpus evaluation loss: %.4f -> %.4f\n", m.Updates(), initial, final)
	for _, prompt := range []string{"the sun was", "the little fish", "user: hello\nfish:"} {
		text, err := m.Generate(ctx, prompt, generation.WithMaxTokens(100), generation.WithTemperature(0))
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%s%s\n", prompt, text)
	}
}

package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/JWset/gofish/generation"
	"github.com/JWset/gofish/llama"
	"github.com/JWset/gofish/training"
)

func main() {
	steps := flag.Int("steps", 2400, "AdamW updates")
	flag.Parse()
	data, err := os.ReadFile("lm/corpora/tiny-stories.txt")
	if err != nil {
		log.Fatal(err)
	}
	m, err := llama.NewByteModel(llama.Config{ContextSize: 32, Width: 32, Heads: 4, KVHeads: 2, Layers: 1, FeedForward: 64, Seed: 42})
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	initial, err := m.Loss(ctx, string(data))
	if err != nil {
		log.Fatal(err)
	}
	start := time.Now()
	fmt.Printf("parameters=%d initial_loss=%.4f\n", m.ParameterCount(), initial)
	_, err = m.Train(ctx, string(data), training.Options{Steps: *steps, BatchSize: 4, LearningRate: 0.003, WeightDecay: 0.01, Seed: 42, OnStep: func(s training.Step) {
		if s.Number%200 == 0 {
			fmt.Printf("step=%d loss=%.4f elapsed=%s\n", s.Number, s.Loss, time.Since(start).Round(time.Second))
		}
	}})
	if err != nil {
		log.Fatal(err)
	}
	final, err := m.Loss(ctx, string(data))
	if err != nil {
		log.Fatal(err)
	}
	if err = m.Save("llama/assets/tiny-llama.gguf"); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("corpus loss %.4f -> %.4f\n", initial, final)
	text, err := m.Generate(ctx, "the little fish", generation.WithTemperature(0), generation.WithMaxTokens(100))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("the little fish" + text)
}

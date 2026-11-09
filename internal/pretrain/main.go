// Command pretrain rebuilds the checked-in demonstration checkpoints. It has no
// dependencies, network requests or external datasets.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/JWset/gofish/lm"
)

func main() {
	steps := flag.Int("steps", 1800, "Adam updates per model")
	flag.Parse()
	for _, name := range []string{"tiny-stories", "tiny-dialogue"} {
		data, err := os.ReadFile(filepath.Join("lm", "corpora", name+".txt"))
		if err != nil {
			log.Fatal(err)
		}
		m, err := lm.New(string(data), lm.Config{ContextSize: 12, EmbeddingSize: 8, HiddenSize: 64, Seed: 42})
		if err != nil {
			log.Fatal(err)
		}
		initial, err := m.Loss(context.Background(), string(data))
		if err != nil {
			log.Fatal(err)
		}
		_, err = m.Train(context.Background(), string(data), lm.TrainOptions{Steps: *steps, BatchSize: 32, LearningRate: 0.01, Seed: 42, OnStep: func(s lm.Step) {
			if s.Number%300 == 0 {
				fmt.Printf("%s step=%d loss=%.4f\n", name, s.Number, s.Loss)
			}
		}})
		if err != nil {
			log.Fatal(err)
		}
		final, err := m.Loss(context.Background(), string(data))
		if err != nil {
			log.Fatal(err)
		}
		if err = m.Save(filepath.Join("lm", "assets", name+".gofish")); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%s evaluation loss: %.4f -> %.4f\n", name, initial, final)
	}
}

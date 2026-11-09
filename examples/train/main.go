package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/JWset/gofish"
)

func main() {
	ctx := context.Background()
	corpus := "hello fish. hello bird. hello fish. hello bird.\n"
	model, err := gofish.NewLanguageModel(corpus, gofish.Config{ContextSize: 4, EmbeddingSize: 4, HiddenSize: 16, Seed: 42})
	if err != nil {
		log.Fatal(err)
	}
	_, err = model.Train(ctx, corpus, gofish.TrainOptions{Steps: 200, BatchSize: 16, Seed: 42})
	if err != nil {
		log.Fatal(err)
	}
	path := "model.gofish"
	if len(os.Args) > 1 {
		path = os.Args[1]
	}
	if err = model.Save(path); err != nil {
		log.Fatal(err)
	}
	model, err = gofish.Load(path)
	if err != nil {
		log.Fatal(err)
	}
	text, err := model.Generate(ctx, "hello", gofish.WithMaxTokens(60), gofish.WithTemperature(0))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("hello" + text)
	fmt.Println("Saved:", path)
}

// Run the bundled trainable byte-level transformer with no network or Python.
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/JWset/gofish"
)

func main() {
	model, err := gofish.Load("tiny-transformer")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Native transformer: %d parameters\n", model.ParameterCount())
	prompt := "the little fish"
	fmt.Print(prompt)
	_, err = model.Generate(context.Background(), prompt, gofish.WithMaxTokens(160), gofish.WithTemperature(0.5), gofish.WithTopK(20), gofish.WithTopP(0.9), gofish.WithSeed(42), gofish.WithStream(func(chunk string) error { _, err := fmt.Print(chunk); return err }))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println()
}

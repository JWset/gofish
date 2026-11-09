package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/JWset/gofish"
)

func main() {
	name := "tiny-llama"
	if len(os.Args) > 1 {
		name = os.Args[1]
	}
	m, err := gofish.Load(name)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Native GGUF: %d parameters\n", m.ParameterCount())
	prompt := "the little fish"
	fmt.Print(prompt)
	_, err = m.Generate(context.Background(), prompt, gofish.WithTemperature(0), gofish.WithMaxTokens(100), gofish.WithStream(func(s string) error { _, err := fmt.Print(s); return err }))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println()
}

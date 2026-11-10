// Run: go run ./examples/local /path/to/Qwen3-0.6B-Q8_0.gguf
package main

import (
	"context"
	"fmt"
	"github.com/JWset/gofish"
	"log"
	"os"
)

func main() {
	if len(os.Args) != 2 {
		log.Fatal("usage: local /path/to/model.gguf; install llama-server or set GOFISH_LLAMA_SERVER")
	}
	ctx := context.Background()
	model, err := gofish.LoadPretrained(ctx, os.Args[1], gofish.GGUFOptions{ContextSize: 2048})
	if err != nil {
		log.Fatal(err)
	}
	defer model.Close()
	_, err = model.Chat(ctx, []gofish.Message{{Role: "user", Content: "Explain what a language model is in two sentences."}}, gofish.WithMaxTokens(100), gofish.WithStream(func(s string) error { _, err := fmt.Print(s); return err }))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println()
}

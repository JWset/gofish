// Install and start Ollama first. This example explicitly downloads the model
// if it is missing; the download may be hundreds of megabytes or more.
package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/JWset/gofish"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	model, err := gofish.Load("ollama:qwen3:0.6b")
	if err != nil {
		log.Fatal(err)
	}
	if err = model.Pull(ctx); err != nil {
		log.Fatal(err)
	}
	_, err = model.Chat(ctx, []gofish.Message{{Role: "user", Content: "Tell me a short story about a fish."}}, gofish.WithMaxTokens(200), gofish.WithStream(func(chunk string) error { _, err := fmt.Print(chunk); return err }))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println()
}

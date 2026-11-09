package main

import (
	"context"
	"fmt"
	"log"

	"github.com/JWset/gofish"
)

func main() {
	model, err := gofish.Load("tiny-stories")
	if err != nil {
		log.Fatal(err)
	}
	text, err := model.Generate(context.Background(), "the sun was", gofish.WithTemperature(0), gofish.WithMaxTokens(100))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("the sun was" + text)
}

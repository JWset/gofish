package llama

import (
	"bytes"
	"embed"

	"github.com/JWset/gofish/gguf"
)

//go:embed assets/*
var assets embed.FS

func LoadBytes(data []byte) (*Model, error) {
	f, err := gguf.Read(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return nil, err
	}
	return fromGGUF(f, LoadOptions{})
}

// Pretrained loads the bundled tiny Llama-shaped model from a real GGUF file.
// Its gofish-byte tokenizer is custom; it is an offline engineering demonstration.
func Pretrained() (*Model, error) {
	data, err := assets.ReadFile("assets/tiny-llama.gguf")
	if err != nil {
		return nil, err
	}
	return LoadBytes(data)
}

package transformer

import "embed"

//go:embed assets/*
var assets embed.FS

// Pretrained loads the bundled tiny-transformer, including trained weights and
// optimizer moments. It is a tiny corpus-memorization demo, not an instruction LLM.
func Pretrained() (*Model, error) {
	data, err := assets.ReadFile("assets/tiny-transformer.gofish")
	if err != nil {
		return nil, err
	}
	return LoadBytes(data)
}

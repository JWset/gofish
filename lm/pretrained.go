package lm

import (
	"embed"
	"fmt"
)

//go:embed assets/*
var assets embed.FS

// Pretrained loads a bundled, genuinely trained native model with no network.
// These tiny models reproduce patterns from their included training corpora.
// They are text-generation demonstrations, not general-purpose assistants.
func Pretrained(name string) (*Model, error) {
	if name != "tiny-stories" && name != "tiny-dialogue" {
		return nil, fmt.Errorf("gofish: unknown native pretrained model %q", name)
	}
	data, err := assets.ReadFile("assets/" + name + ".gofish")
	if err != nil {
		return nil, err
	}
	return LoadBytes(data)
}

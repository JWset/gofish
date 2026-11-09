package gofish

type ModelInfo struct {
	Name        string
	Description string
	Bundled     bool
	Trainable   bool
}

// Models lists starter models. Any installed Ollama model is also accepted via
// the ollama: prefix, beyond the two suggested pretrained sizes listed here.
func Models() []ModelInfo {
	return []ModelInfo{
		{Name: "tiny-transformer", Description: "Bundled byte-level causal transformer; trainable CPU demonstration", Bundled: true, Trainable: true},
		{Name: "tiny-stories", Description: "Bundled character model trained on a short fish story; demonstration only", Bundled: true, Trainable: true},
		{Name: "tiny-dialogue", Description: "Bundled character model trained on a handful of dialogue examples; demonstration only", Bundled: true, Trainable: true},
		{Name: "ollama:qwen3:0.6b", Description: "Small pretrained instruction model; requires Ollama and an explicit download"},
		{Name: "ollama:qwen3:4b", Description: "Larger pretrained instruction model; requires Ollama and an explicit download"},
	}
}

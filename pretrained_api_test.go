package gofish_test

import (
	"context"
	"errors"
	"github.com/JWset/gofish"
	"os"
	"testing"
	"time"
)

func TestManagedPretrainedPublicAPI(t *testing.T) {
	path := os.Getenv("GOFISH_QWEN_GGUF")
	if path == "" {
		t.Skip("set GOFISH_QWEN_GGUF for public API integration")
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	model, err := gofish.LoadGGUFContext(ctx, path, gofish.GGUFOptions{ContextSize: 512, Threads: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	if !model.Capabilities().Chat || model.Capabilities().Train || model.ParameterCount() == 0 {
		t.Fatal("wrong pretrained capabilities")
	}
	if _, err = model.Train(ctx, "hello", gofish.TrainOptions{}); !errors.Is(err, gofish.ErrUnsupported) {
		t.Fatal(err)
	}
	if err = model.Save(t.TempDir() + "/copy.gguf"); !errors.Is(err, gofish.ErrUnsupported) {
		t.Fatal(err)
	}
	info, err := model.PretrainedInfo()
	if err != nil || info.Architecture != "qwen3" {
		t.Fatalf("%+v %v", info, err)
	}
	messages := []gofish.Message{{Role: "user", Content: "Say hello."}}
	if _, err = model.Chat(ctx, messages, gofish.WithMaxTokens(16)); err != nil {
		t.Fatal(err)
	}
	ids, err := model.Tokenize(ctx, "hello")
	if err != nil {
		t.Fatal(err)
	}
	if text, err := model.Detokenize(ctx, ids); err != nil || text != "hello" {
		t.Fatalf("%q %v", text, err)
	}
	if _, err = model.ApplyChatTemplate(ctx, messages); err != nil {
		t.Fatal(err)
	}
	if err = model.Close(); err != nil {
		t.Fatal(err)
	}
	if err = model.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestPretrainedCatalog(t *testing.T) {
	seen := map[string]bool{}
	for _, p := range gofish.PretrainedModels() {
		if seen[p.Name] || len(p.Revision) != 40 || len(p.SHA256) != 64 {
			t.Fatalf("invalid preset %+v", p)
		}
		seen[p.Name] = true
	}
	for _, name := range []string{"qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "gemma", "gemma2", "gemma3", "smollm1", "smollm2", "smollm3"} {
		if !seen[name] {
			t.Fatalf("missing preset %s", name)
		}
	}
}

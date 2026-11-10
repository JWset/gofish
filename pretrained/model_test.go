package pretrained

import (
	"bytes"
	"errors"
	"github.com/JWset/gofish/gguf"
	"os"
	"path/filepath"
	"testing"
)

func metadataFile(t *testing.T, arch, tok string, extra map[string]any) string {
	t.Helper()
	meta := map[string]any{"general.architecture": arch, "general.name": "SmolLM fixture", "tokenizer.ggml.model": tok, arch + ".context_length": uint32(4096), arch + ".embedding_length": uint32(8), arch + ".attention.head_count": uint32(2), arch + ".block_count": uint32(1)}
	for k, v := range extra {
		meta[k] = v
	}
	var buf bytes.Buffer
	if err := gguf.Write(&buf, meta, []gguf.TensorData{{Name: "token_embd.weight", Dimensions: []uint64{8, 1}, Type: gguf.F32, Data: make([]byte, 32)}}); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "fixture.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	return path
}
func TestArchitectureContracts(t *testing.T) {
	for _, c := range []struct{ arch, tok string }{{"qwen3", "gpt2"}, {"gemma", "llama"}, {"gemma2", "llama"}, {"gemma3", "llama"}, {"llama", "gpt2"}, {"smollm3", "gpt2"}} {
		t.Run(c.arch, func(t *testing.T) {
			p := metadataFile(t, c.arch, c.tok, nil)
			info, err := Inspect(p, Options{ContextSize: 1024})
			if err != nil {
				t.Fatal(err)
			}
			if info.ContextSize != 1024 || info.Architecture != c.arch {
				t.Fatalf("%+v", info)
			}
			if _, err = Inspect(p, Options{ContextSize: 8192}); err == nil {
				t.Fatal("accepted context beyond training limit")
			}
			if _, err = Inspect(p, Options{MaxModelBytes: 1}); err == nil {
				t.Fatal("ignored file budget")
			}
			if _, err = Inspect(p, Options{MaxCacheBytes: 1}); err == nil {
				t.Fatal("ignored cache budget")
			}
		})
	}
	p := metadataFile(t, "qwen3moe", "gpt2", nil)
	if _, err := Inspect(p, Options{}); !errors.Is(err, ErrArchitecture) {
		t.Fatal(err)
	}
	p = metadataFile(t, "qwen3", "llama", nil)
	if _, err := Inspect(p, Options{}); err == nil {
		t.Fatal("accepted wrong tokenizer")
	}
	p = metadataFile(t, "qwen3", "gpt2", map[string]any{"split.count": uint32(2)})
	if _, err := Inspect(p, Options{}); err == nil {
		t.Fatal("accepted split weights")
	}
}

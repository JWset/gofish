package pretrained

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/JWset/gofish/chat"
	"github.com/JWset/gofish/generation"
)

// Every row is a separate compatibility target. No downloads occur in tests.
// See PRETRAINED.md for the real-checkpoint manifest and opt-in invocation.
func TestGGUFCompatibility(t *testing.T) {
	var paths map[string]string
	if os.Getenv("GOFISH_GGUF_MODELS") == "" && os.Getenv("GOFISH_GGUF_DIR") == "" {
		t.Skip("set GOFISH_GGUF_DIR or GOFISH_GGUF_MODELS for real checkpoint tests")
	}
	if value := os.Getenv("GOFISH_GGUF_MODELS"); value != "" {
		if err := json.Unmarshal([]byte(value), &paths); err != nil {
			t.Fatal(err)
		}
	}
	if paths == nil {
		paths = map[string]string{}
	}
	cases := Models()
	if dir := os.Getenv("GOFISH_GGUF_DIR"); dir != "" {
		for _, c := range cases {
			if paths[c.Name] == "" {
				path := filepath.Join(dir, c.Filename)
				if _, err := os.Stat(path); err == nil {
					paths[c.Name] = path
				}
			}
		}
	}
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			path := paths[c.Name]
			if path == "" {
				if os.Getenv("GOFISH_REQUIRE_ALL_GGUF") == "1" {
					t.Fatal("required checkpoint is missing")
				}
				t.Skip("no checkpoint supplied for this architecture/version")
			}
			if os.Getenv("GOFISH_VERIFY_GGUF_SHA") == "1" {
				f, err := os.Open(path)
				if err != nil {
					t.Fatal(err)
				}
				hash := sha256.New()
				_, err = io.Copy(hash, f)
				f.Close()
				if err != nil {
					t.Fatal(err)
				}
				if fmt.Sprintf("%x", hash.Sum(nil)) != c.SHA256 {
					t.Fatal("checkpoint SHA-256 differs from pinned compatibility manifest")
				}
			}
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
			defer cancel()
			m, err := Load(ctx, path, Options{ContextSize: 512, Threads: 4})
			if err != nil {
				t.Fatal(err)
			}
			defer m.Close()
			info := m.Info()
			if info.Architecture != c.Architecture || info.Family != c.Family {
				t.Fatalf("wrong architecture/family: %+v", info)
			}
			if len(info.Quantizations) == 0 {
				t.Fatal("missing quantization descriptors")
			}
			text := "Hello, world! café 中文 🐟"
			ids, err := m.Tokenize(ctx, text)
			if err != nil || len(ids) == 0 {
				t.Fatalf("tokenize: %v", err)
			}
			decoded, err := m.Detokenize(ctx, ids)
			if err != nil || (decoded != text && !(info.TokenizerAddsSpacePrefix && decoded == " "+text)) {
				t.Fatalf("tokenizer roundtrip %q -> %q: %v", text, decoded, err)
			}
			messages := []chat.Message{{Role: "user", Content: "Reply with just the number: 2 + 2 ="}}
			prompt, err := m.ApplyChatTemplate(ctx, messages)
			if err != nil || !strings.Contains(prompt, messages[0].Content) {
				t.Fatalf("template: %q %v", prompt, err)
			}
			var chunks strings.Builder
			result, err := m.Chat(ctx, messages, generation.WithMaxTokens(32), generation.WithTemperature(0), generation.WithStream(func(s string) error { chunks.WriteString(s); return nil }))
			if err != nil || strings.TrimSpace(result) == "" || result != chunks.String() {
				t.Fatalf("chat stream: %q %v", result, err)
			}
			var rawChunks strings.Builder
			raw, err := m.Generate(ctx, "The capital of France is", generation.WithMaxTokens(8), generation.WithTemperature(0), generation.WithStream(func(s string) error { rawChunks.WriteString(s); return nil }))
			if err != nil || raw == "" || raw != rawChunks.String() {
				t.Fatalf("raw generation: %q %v", raw, err)
			}
			schema := json.RawMessage(`{"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"],"additionalProperties":false}`)
			var jsonChunks strings.Builder
			resultJSON, err := m.ChatStructured(ctx, messages, chat.Options{ResponseSchema: schema}, generation.WithMaxTokens(64), generation.WithTemperature(0), generation.WithStream(func(s string) error { jsonChunks.WriteString(s); return nil }))
			if err != nil || !json.Valid([]byte(resultJSON.Message.Content)) || resultJSON.Message.Content != jsonChunks.String() {
				t.Fatalf("structured JSON: %+v %v", resultJSON, err)
			}
			{
				var deltas int
				tools := chat.Options{Tools: []chat.Tool{{Type: "function", Function: chat.Function{Name: "get_weather", Description: "Get weather in a city", Parameters: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}`), Strict: true}}}, ToolChoice: "required", OnDelta: func(d chat.Delta) error { deltas += len(d.ToolCalls); return nil }}
				response, err := m.ChatStructured(ctx, []chat.Message{{Role: "user", Content: "Call get_weather for Tokyo."}}, tools, generation.WithMaxTokens(128), generation.WithTemperature(0))
				if err != nil || len(response.Message.ToolCalls) != 1 || deltas == 0 {
					t.Fatalf("function calling: %+v %v", response, err)
				}
				call := response.Message.ToolCalls[0]
				var args map[string]any
				if json.Unmarshal([]byte(call.Function.Arguments), &args) != nil || args["city"] == nil {
					t.Fatalf("invalid structured arguments: %+v", call)
				}
				history := []chat.Message{{Role: "user", Content: "What is the weather in Tokyo?"}, response.Message, {Role: "tool", ToolCallID: call.ID, Content: `{"temperature":22,"conditions":"sunny"}`}}
				if _, err = m.Chat(ctx, history, generation.WithMaxTokens(32)); err != nil {
					t.Fatalf("tool result conversation: %v", err)
				}
			}
			t.Logf("verified %s context=%d quantization=%v tokenizer=%s", c.Name, info.ContextSize, info.Quantizations, info.Tokenizer)
			if err = m.Close(); err != nil {
				t.Fatal(err)
			}
			if _, err = m.Tokenize(ctx, "closed"); err != ErrClosed {
				t.Fatalf("closed runtime: %v", err)
			}
		})
	}
}

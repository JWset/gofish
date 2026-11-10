package pretrained

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/JWset/gofish/chat"
)

func TestJSONToolsUseSchemaAndPreserveHistory(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
		}
		if request["tools"] != nil || request["response_format"] == nil {
			t.Error("JSON tools must use constrained output, not native tool syntax")
		}
		content := `{"content":"","tool_calls":[{"name":"weather","arguments":{"city":"Tokyo"}}]}`
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"message": chat.Message{Role: "assistant", Content: content}, "finish_reason": "stop"}}})
	}))
	defer server.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	m := &Model{info: Info{Family: "gemma3", HasChatTemplate: true}, url: server.URL, client: server.Client(), ctx: ctx, cancel: cancel, done: make(chan struct{})}
	history := []chat.Message{{Role: "user", Content: "weather in Tokyo?"}}
	calls := 0
	options := chat.Options{Tools: []chat.Tool{{Type: "function", Function: chat.Function{Name: "weather", Parameters: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`)}}}, ToolChoice: "required", OnDelta: func(delta chat.Delta) error { calls += len(delta.ToolCalls); return nil }}
	result, err := m.ChatStructured(ctx, history, options)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 || result.FinishReason != "tool_calls" || result.Message.ToolCalls[0].Function.Arguments != `{"city":"Tokyo"}` {
		t.Fatalf("%+v calls=%d", result, calls)
	}
	if history[0].Content != "weather in Tokyo?" {
		t.Fatal("mutated caller history")
	}
	tool := chat.Message{Role: "tool", ToolCallID: result.Message.ToolCalls[0].ID, Content: `{"temperature":22}`}
	conversation := []chat.Message{history[0], result.Message, tool}
	normalized := normalizeToolHistory(conversation)
	if normalized[2].Role != "user" || !strings.Contains(normalized[1].Content, `"name":"weather"`) {
		t.Fatalf("%+v", normalized)
	}
	if conversation[2].Role != "tool" || len(conversation[1].ToolCalls) != 1 {
		t.Fatal("mutated caller tool history")
	}
}

func TestGemmaOneSchemaUsesTemplatedRawStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		switch r.URL.Path {
		case "/apply-template":
			_ = json.NewEncoder(w).Encode(map[string]any{"prompt": "<start_of_turn>user\nhello<end_of_turn>\n<start_of_turn>model\n"})
		case "/v1/completions":
			if body["json_schema"] == nil || !strings.Contains(body["prompt"].(string), "<start_of_turn>") {
				t.Error("lost schema or model template")
			}
			w.Header().Set("Content-Type", "text/event-stream")
			w.Write([]byte("data: {\"choices\":[{\"text\":\"{\\\"answer\\\":4}\",\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
		default:
			t.Errorf("unexpected parser endpoint %s", r.URL.Path)
			http.Error(w, "unexpected", 400)
		}
	}))
	defer server.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	m := &Model{info: Info{Architecture: "gemma", Family: "gemma", HasChatTemplate: true}, url: server.URL, client: server.Client(), ctx: ctx, cancel: cancel, done: make(chan struct{})}
	chunks := ""
	response, err := m.ChatStructured(ctx, []chat.Message{{Role: "user", Content: "hello"}}, chat.Options{ResponseSchema: json.RawMessage(`{"type":"object"}`), OnDelta: func(d chat.Delta) error { chunks += d.Content; return nil }})
	if err != nil || response.Message.Content != `{"answer":4}` || chunks != response.Message.Content {
		t.Fatalf("%+v %q %v", response, chunks, err)
	}
}

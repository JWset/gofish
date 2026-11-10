package pretrained

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/JWset/gofish/chat"
	"github.com/JWset/gofish/generation"
)

func TestSSEBoundaries(t *testing.T) {
	for _, c := range []struct {
		name, body string
		wantErr    bool
	}{
		{"multiline", ": keepalive\ndata: {\ndata: \"value\":1}\n\ndata: [DONE]\n\n", false},
		{"truncated", "data: {\"value\":1}\n\n", true},
		{"invalidJSON", "data: nope\n\ndata: [DONE]\n\n", false},
	} {
		t.Run(c.name, func(t *testing.T) {
			count := 0
			err := readEvents(strings.NewReader(c.body), func(data []byte) error { count++; return nil })
			if (err != nil) != c.wantErr {
				t.Fatal(err)
			}
			if count != 1 {
				t.Fatal(count)
			}
		})
	}
	sentinel := errors.New("stop")
	if err := readEvents(strings.NewReader("data: {}\n\n"), func([]byte) error { return sentinel }); !errors.Is(err, sentinel) {
		t.Fatal(err)
	}
}
func TestStructuredStreamingAndCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer secret" {
			t.Error("missing authentication")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call1\",\"type\":\"function\",\"function\":{\"name\":\"weather\",\"arguments\":\"{\\\"city\\\":\"}}]}}]}\n\n")
		io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"Tokyo\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\ndata: [DONE]\n\n")
	}))
	defer server.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	m := &Model{info: Info{HasChatTemplate: true}, url: server.URL, key: "secret", client: server.Client(), ctx: ctx, cancel: cancel, done: make(chan struct{})}
	tools := chat.Options{Tools: []chat.Tool{{Type: "function", Function: chat.Function{Name: "weather", Parameters: []byte(`{"type":"object"}`)}}}, ToolChoice: "required", OnDelta: func(chat.Delta) error { return nil }}
	response, err := m.ChatStructured(context.Background(), []chat.Message{{Role: "user", Content: "weather?"}}, tools)
	if err != nil {
		t.Fatal(err)
	}
	if response.Message.ToolCalls[0].Function.Arguments != `{"city":"Tokyo"}` {
		t.Fatalf("%+v", response)
	}
	sentinel := errors.New("cancel callback")
	tools.OnDelta = func(chat.Delta) error { return sentinel }
	if _, err = m.ChatStructured(context.Background(), []chat.Message{{Role: "user", Content: "weather?"}}, tools); !errors.Is(err, sentinel) {
		t.Fatal(err)
	}
	canceled, c := context.WithCancel(context.Background())
	c()
	if _, err = m.Generate(canceled, "prompt", generation.WithMaxTokens(1)); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	m.Close()
	if _, err = m.Generate(context.Background(), "prompt"); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
}
func TestRejectInvalidTools(t *testing.T) {
	message := []chat.Message{{Role: "user", Content: "hello"}}
	for _, o := range []chat.Options{{ToolChoice: "required"}, {ToolChoice: "surprise"}, {Tools: []chat.Tool{{Type: "other"}}}, {Tools: []chat.Tool{{Type: "function", Function: chat.Function{Name: "bad", Parameters: []byte(`{"type":"array"}`)}}}}} {
		if validateChat(message, o) == nil {
			t.Fatal("accepted invalid tools")
		}
	}
	if validateResult(chat.Response{Message: chat.Message{ToolCalls: []chat.ToolCall{{ID: "id", Type: "function", Function: chat.FunctionCall{Name: "undeclared", Arguments: "{}"}}}}}, chat.Options{}) == nil {
		t.Fatal("accepted undeclared function")
	}
}

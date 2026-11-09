package ollama_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/JWset/gofish/generation"
	"github.com/JWset/gofish/ollama"
)

// Exercise real JSON requests and response decoding without sockets, which are
// unavailable in restricted build environments. The transport invokes a handler.
type transport func(*http.Request) (*http.Response, error)

func (f transport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

type testServer struct {
	URL    string
	Client *http.Client
}

func (s *testServer) Close() {}
func newServer(handler http.Handler) *testServer {
	return &testServer{URL: "http://ollama.test", Client: &http.Client{Transport: transport(func(r *http.Request) (*http.Response, error) {
		if err := r.Context().Err(); err != nil {
			return nil, err
		}
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, r)
		return w.Result(), nil
	})}}
}

func TestGenerateChatPullAndStreaming(t *testing.T) {
	server := newServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" || r.Header.Get("Content-Type") != "application/json" {
			t.Error("wrong request headers")
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			return
		}
		if body["model"] != "qwen3:0.6b" {
			t.Error("model not passed")
		}
		switch r.URL.Path {
		case "/api/pull":
			if body["stream"] != false {
				t.Error("pull must use nonstreaming response")
			}
			fmt.Fprint(w, `{"status":"success"}`)
		case "/api/chat":
			if len(body["messages"].([]any)) != 2 {
				t.Error("chat history missing")
			}
			fmt.Fprint(w, `{"message":{"role":"assistant","content":"hello fish"},"done":true}`)
		case "/api/generate":
			options := body["options"].(map[string]any)
			if options["num_predict"] != float64(12) {
				t.Error("generation options missing")
			}
			if options["top_k"] != float64(7) || options["top_p"] != 0.9 {
				t.Error("sampling filters missing")
			}
			if body["stream"] == true {
				fmt.Fprintln(w, `{"response":"hello ","done":false}`)
				fmt.Fprintln(w, `{"response":"fish","done":true}`)
			} else {
				fmt.Fprint(w, `{"response":"hello fish","done":true}`)
			}
		default:
			t.Error("unexpected endpoint")
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	m, err := ollama.New("qwen3:0.6b", ollama.WithBaseURL(server.URL), ollama.WithHTTPClient(server.Client))
	if err != nil {
		t.Fatal(err)
	}
	if err = m.Pull(context.Background()); err != nil {
		t.Fatal(err)
	}
	options := []generation.Option{generation.WithMaxTokens(12), generation.WithTopK(7), generation.WithTopP(0.9)}
	text, err := m.Generate(context.Background(), "hello", options...)
	if err != nil || text != "hello fish" {
		t.Fatalf("%q, %v", text, err)
	}
	var streamed strings.Builder
	options = append(options, generation.WithStream(func(s string) error { streamed.WriteString(s); return nil }))
	text, err = m.Generate(context.Background(), "hello", options...)
	if err != nil || text != streamed.String() || text != "hello fish" {
		t.Fatalf("stream %q: %v", text, err)
	}
	text, err = m.Chat(context.Background(), []ollama.Message{{Role: "system", Content: "be brief"}, {Role: "user", Content: "hello"}})
	if err != nil || text != "hello fish" {
		t.Fatal(text, err)
	}
}

func TestErrorsAndCancellation(t *testing.T) {
	for _, tt := range []struct {
		name   string
		status int
		body   string
		want   string
	}{
		{"missing model", 404, `{"error":"model not found"}`, "model not found"},
		{"stream error", 200, `{"error":"out of memory"}`, "out of memory"},
		{"truncated", 200, "{", "incomplete"},
		{"missing done", 200, `{"response":"hello"}`, "incomplete"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			server := newServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(tt.status); fmt.Fprint(w, tt.body) }))
			defer server.Close()
			m, err := ollama.New("test", ollama.WithBaseURL(server.URL), ollama.WithHTTPClient(server.Client))
			if err != nil {
				t.Fatal(err)
			}
			_, err = m.Generate(context.Background(), "hello")
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatal(err)
			}
			if tt.status == 404 {
				var apiError *ollama.APIError
				if !errors.As(err, &apiError) || apiError.StatusCode != 404 {
					t.Fatal("lost status")
				}
			}
		})
	}
	server := newServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, `{"response":"hello","done":false}`)
		w.(http.Flusher).Flush()
	}))
	defer server.Close()
	m, err := ollama.New("test", ollama.WithBaseURL(server.URL), ollama.WithHTTPClient(server.Client))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	_, err = m.Generate(ctx, "hello", generation.WithStream(func(string) error { cancel(); return context.Canceled }))
	if !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err = ollama.New("test", ollama.WithBaseURL("file:///tmp")); err == nil {
		t.Fatal("invalid host accepted")
	}
}

func TestTruncatedStreamPreservesPartialOutput(t *testing.T) {
	server := newServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { fmt.Fprintln(w, `{"response":"hello","done":false}`) }))
	defer server.Close()
	m, err := ollama.New("test", ollama.WithBaseURL(server.URL), ollama.WithHTTPClient(server.Client))
	if err != nil {
		t.Fatal(err)
	}
	text, err := m.Generate(context.Background(), "hello")
	if text != "hello" || !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatal(text, err)
	}
}

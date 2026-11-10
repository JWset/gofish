// Package ollama runs pretrained language models through a separately installed
// Ollama server. It provides inference, streaming and explicit model downloads;
// it does not implement fine-tuning of Ollama models.
package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/JWset/gofish/chat"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/JWset/gofish/generation"
)

type Model struct {
	name    string
	baseURL string
	client  *http.Client
}
type Option func(*Model)

func WithBaseURL(baseURL string) Option {
	return func(m *Model) { m.baseURL = strings.TrimRight(baseURL, "/") }
}
func WithHTTPClient(client *http.Client) Option { return func(m *Model) { m.client = client } }

// New creates a client; no network request or model download occurs until called.
// Use a context deadline to bound requests and model downloads.
func New(name string, options ...Option) (*Model, error) {
	m := &Model{name: name, baseURL: "http://localhost:11434", client: http.DefaultClient}
	for _, option := range options {
		if option == nil {
			return nil, fmt.Errorf("gofish: nil Ollama option")
		}
		option(m)
	}
	u, err := url.Parse(m.baseURL)
	if err != nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") || u.RawQuery != "" || u.Fragment != "" || u.User != nil {
		return nil, fmt.Errorf("gofish: invalid Ollama base URL")
	}
	if strings.TrimSpace(name) == "" || m.client == nil {
		return nil, fmt.Errorf("gofish: model name and HTTP client are required")
	}
	return m, nil
}
func (m *Model) Name() string { return m.name }

// APIError preserves HTTP status and the Ollama error message.
type APIError struct {
	StatusCode int
	Message    string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("gofish: Ollama HTTP %d: %s", e.StatusCode, e.Message)
}

func (m *Model) post(ctx context.Context, path string, body any) (*http.Response, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.baseURL+path, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	response, err := m.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("gofish: connect to Ollama at %s (is ollama serve running?): %w", m.baseURL, err)
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		defer response.Body.Close()
		data, _ := io.ReadAll(io.LimitReader(response.Body, 8192))
		var payload struct {
			Error string `json:"error"`
		}
		_ = json.Unmarshal(data, &payload)
		message := payload.Error
		if message == "" {
			message = strings.TrimSpace(string(data))
		}
		if message == "" {
			message = response.Status
		}
		return nil, &APIError{response.StatusCode, message}
	}
	return response, nil
}

// Pull downloads model weights through Ollama. This can transfer gigabytes;
// callers initiate downloads explicitly, and can cancel through ctx.
func (m *Model) Pull(ctx context.Context) error {
	response, err := m.post(ctx, "/api/pull", map[string]any{"model": m.name, "stream": false})
	if err != nil {
		return err
	}
	defer response.Body.Close()
	var result struct {
		Status string `json:"status"`
		Error  string `json:"error"`
	}
	if err = json.NewDecoder(io.LimitReader(response.Body, 1<<20)).Decode(&result); err != nil {
		return err
	}
	if result.Error != "" {
		return errors.New("gofish: Ollama: " + result.Error)
	}
	if result.Status != "success" {
		return fmt.Errorf("gofish: Ollama pull did not complete: %q", result.Status)
	}
	return nil
}

type Message = chat.Message

func runtimeOptions(c generation.Config) map[string]any {
	options := map[string]any{"num_predict": c.MaxTokens, "temperature": c.Temperature, "seed": c.Seed, "stop": c.Stop, "top_p": c.TopP}
	if c.TopK > 0 {
		options["top_k"] = c.TopK
	}
	return options
}
func (m *Model) Generate(ctx context.Context, prompt string, options ...generation.Option) (string, error) {
	c, err := generation.Resolve(options)
	if err != nil {
		return "", err
	}
	return m.generate(ctx, "/api/generate", map[string]any{"model": m.name, "prompt": prompt, "stream": c.OnToken != nil, "think": false, "options": runtimeOptions(c)}, c)
}

// Chat takes explicit history; it does not retain conversation state on the client.
func (m *Model) Chat(ctx context.Context, messages []Message, options ...generation.Option) (string, error) {
	c, err := generation.Resolve(options)
	if err != nil {
		return "", err
	}
	if len(messages) == 0 {
		return "", fmt.Errorf("gofish: chat requires messages")
	}
	for _, message := range messages {
		if message.Role != "system" && message.Role != "user" && message.Role != "assistant" {
			return "", fmt.Errorf("gofish: unsupported chat role %q", message.Role)
		}
	}
	return m.generate(ctx, "/api/chat", map[string]any{"model": m.name, "messages": messages, "stream": c.OnToken != nil, "think": false, "options": runtimeOptions(c)}, c)
}
func (m *Model) generate(ctx context.Context, path string, body any, c generation.Config) (string, error) {
	response, err := m.post(ctx, path, body)
	if err != nil {
		return "", err
	}
	defer response.Body.Close()
	decoder := json.NewDecoder(io.LimitReader(response.Body, 64<<20))
	collector := generation.NewCollector(c)
	for {
		var event struct {
			Response string  `json:"response"`
			Message  Message `json:"message"`
			Done     bool    `json:"done"`
			Error    string  `json:"error"`
		}
		if err = decoder.Decode(&event); err != nil {
			if errors.Is(err, io.EOF) {
				err = io.ErrUnexpectedEOF
			}
			return collector.Text(), fmt.Errorf("gofish: incomplete Ollama response: %w", err)
		}
		if event.Error != "" {
			return collector.Text(), errors.New("gofish: Ollama: " + event.Error)
		}
		chunk := event.Response
		if path == "/api/chat" {
			chunk = event.Message.Content
		}
		stopped, err := collector.Feed(chunk)
		if err != nil {
			return collector.Text(), err
		}
		if stopped {
			return collector.Text(), nil
		}
		if event.Done {
			err = collector.Finish()
			return collector.Text(), err
		}
	}
}

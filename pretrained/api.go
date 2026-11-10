package pretrained

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/JWset/gofish/chat"
	"github.com/JWset/gofish/generation"
)

func (m *Model) request(ctx context.Context, path string, body any) (*http.Response, error) {
	if err := m.ctx.Err(); err != nil {
		return nil, ErrClosed
	}
	select {
	case <-m.done:
		return nil, fmt.Errorf("gofish: local runtime exited: %v; %s", m.waitErr, m.log.String())
	default:
	}
	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(ctx)
	stop := context.AfterFunc(m.ctx, cancel)
	req, err := http.NewRequestWithContext(ctx, "POST", m.url+path, bytes.NewReader(data))
	if err != nil {
		stop()
		cancel()
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.key)
	response, err := m.client.Do(req)
	if err != nil {
		stop()
		cancel()
		return nil, err
	}
	response.Body = &cancelBody{ReadCloser: response.Body, cancel: func() { stop(); cancel() }}
	if response.StatusCode != 200 {
		defer response.Body.Close()
		data, _ := io.ReadAll(io.LimitReader(response.Body, 8192))
		return nil, fmt.Errorf("gofish: local runtime %s returned HTTP %d: %s; runtime: %s", path, response.StatusCode, data, m.log.String())
	}
	return response, nil
}

type cancelBody struct {
	io.ReadCloser
	cancel func()
}

func (b *cancelBody) Close() error { err := b.ReadCloser.Close(); b.cancel(); return err }
func decode(response *http.Response, out any) error {
	defer response.Body.Close()
	return json.NewDecoder(io.LimitReader(response.Body, 16<<20)).Decode(out)
}
func sampling(c generation.Config) map[string]any {
	return map[string]any{"model": "gofish-local", "max_tokens": c.MaxTokens, "temperature": c.Temperature, "top_k": c.TopK, "top_p": c.TopP, "seed": c.Seed, "stop": c.Stop}
}

func (m *Model) Tokenize(ctx context.Context, text string) ([]int, error) {
	var result struct {
		Tokens []int `json:"tokens"`
	}
	response, err := m.request(ctx, "/tokenize", map[string]any{"content": text, "add_special": false, "parse_special": false})
	if err == nil {
		err = decode(response, &result)
	}
	return result.Tokens, err
}
func (m *Model) Detokenize(ctx context.Context, tokens []int) (string, error) {
	var result struct {
		Content string `json:"content"`
	}
	response, err := m.request(ctx, "/detokenize", map[string]any{"tokens": tokens})
	if err == nil {
		err = decode(response, &result)
	}
	return result.Content, err
}
func (m *Model) ApplyChatTemplate(ctx context.Context, messages []chat.Message) (string, error) {
	if !m.info.HasChatTemplate {
		return "", fmt.Errorf("gofish: GGUF has no conversation template; set GGUFOptions.ChatTemplate to a Jinja template file")
	}
	var result struct {
		Prompt string `json:"prompt"`
	}
	response, err := m.request(ctx, "/apply-template", map[string]any{"messages": messages, "add_generation_prompt": true, "chat_template_kwargs": map[string]any{"enable_thinking": false}})
	if err == nil {
		err = decode(response, &result)
	}
	return result.Prompt, err
}
func (m *Model) Generate(ctx context.Context, prompt string, options ...generation.Option) (string, error) {
	c, err := generation.Resolve(options)
	if err != nil {
		return "", err
	}
	body := sampling(c)
	body["prompt"] = prompt
	body["stream"] = c.OnToken != nil
	response, err := m.request(ctx, "/v1/completions", body)
	if err != nil {
		return "", err
	}
	if c.OnToken == nil {
		var result struct {
			Choices []struct {
				Text string `json:"text"`
			} `json:"choices"`
		}
		if err = decode(response, &result); err != nil {
			return "", err
		}
		if len(result.Choices) != 1 {
			return "", fmt.Errorf("gofish: runtime returned no completion")
		}
		return result.Choices[0].Text, nil
	}
	defer response.Body.Close()
	var text strings.Builder
	err = readEvents(response.Body, func(data []byte) error {
		var event wireResponse
		if e := json.Unmarshal(data, &event); e != nil {
			return e
		}
		if event.Error != nil && string(event.Error) != "null" {
			return fmt.Errorf("gofish: runtime stream error: %s", event.Error)
		}
		for _, choice := range event.Choices {
			text.WriteString(choice.Text)
			if choice.Text != "" {
				if e := c.OnToken(choice.Text); e != nil {
					return e
				}
			}
		}
		return nil
	})
	return text.String(), err
}
func (m *Model) Chat(ctx context.Context, messages []chat.Message, options ...generation.Option) (string, error) {
	result, err := m.ChatStructured(ctx, messages, chat.Options{}, options...)
	return result.Message.Content, err
}

type wireResponse struct {
	Choices []struct {
		Text         string       `json:"text"`
		Message      chat.Message `json:"message"`
		Delta        chat.Delta   `json:"delta"`
		FinishReason string       `json:"finish_reason"`
	} `json:"choices"`
	Usage chat.Usage      `json:"usage"`
	Error json.RawMessage `json:"error"`
}

func validateChat(messages []chat.Message, o chat.Options) error {
	if o.ToolMode != "" && o.ToolMode != "auto" && o.ToolMode != "native" && o.ToolMode != "json" {
		return fmt.Errorf("gofish: ToolMode must be auto, native, or json")
	}
	if len(messages) == 0 {
		return fmt.Errorf("gofish: conversation must contain a message")
	}
	calls := map[string]bool{}
	for _, m := range messages {
		switch m.Role {
		case "system", "user", "assistant", "tool":
		default:
			return fmt.Errorf("gofish: unsupported conversation role %q", m.Role)
		}
		for _, call := range m.ToolCalls {
			if m.Role != "assistant" || call.ID == "" || calls[call.ID] || call.Type != "function" || call.Function.Name == "" || !json.Valid([]byte(call.Function.Arguments)) {
				return fmt.Errorf("gofish: invalid assistant tool-call history")
			}
			calls[call.ID] = true
		}
		if m.Role == "tool" && !calls[m.ToolCallID] {
			return fmt.Errorf("gofish: tool reply has no matching earlier assistant call %q", m.ToolCallID)
		}
		if m.Role == "tool" && m.ToolCallID == "" {
			return fmt.Errorf("gofish: tool reply requires ToolCallID")
		}
	}
	seen := map[string]bool{}
	for _, t := range o.Tools {
		if t.Type != "function" || t.Function.Name == "" || seen[t.Function.Name] {
			return fmt.Errorf("gofish: tool must have type function and a unique nonempty name")
		}
		var schema map[string]any
		if json.Unmarshal(t.Function.Parameters, &schema) != nil || schema["type"] != "object" {
			return fmt.Errorf("gofish: function %s requires an object JSON schema", t.Function.Name)
		}
		seen[t.Function.Name] = true
	}
	switch o.ToolChoice {
	case "", "auto", "none":
	case "required":
		if len(o.Tools) == 0 {
			return fmt.Errorf("gofish: required tool choice needs tools")
		}
	default:
		return fmt.Errorf("gofish: ToolChoice must be auto, none, or required")
	}
	if len(o.ResponseSchema) > 0 && (!json.Valid(o.ResponseSchema) || len(o.Tools) > 0) {
		return fmt.Errorf("gofish: ResponseSchema must be valid JSON and cannot be combined with tools")
	}
	return nil
}
func validateResult(result chat.Response, o chat.Options) error {
	if o.ToolChoice == "none" && len(result.Message.ToolCalls) > 0 {
		return fmt.Errorf("gofish: model returned a function call despite ToolChoice none")
	}
	if o.ToolChoice == "required" && len(result.Message.ToolCalls) == 0 {
		return fmt.Errorf("gofish: model returned no function call despite required ToolChoice")
	}
	for _, call := range result.Message.ToolCalls {
		found := false
		for _, tool := range o.Tools {
			if tool.Function.Name == call.Function.Name {
				found = true
				break
			}
		}
		var arguments map[string]json.RawMessage
		if !found || call.ID == "" || call.Type != "function" || json.Unmarshal([]byte(call.Function.Arguments), &arguments) != nil || arguments == nil {
			return fmt.Errorf("gofish: model returned invalid or undeclared function call %q", call.Function.Name)
		}
	}
	if len(o.ResponseSchema) > 0 && !json.Valid([]byte(result.Message.Content)) {
		return fmt.Errorf("gofish: model returned invalid structured JSON")
	}
	return nil
}
func (m *Model) ChatStructured(ctx context.Context, messages []chat.Message, o chat.Options, options ...generation.Option) (result chat.Response, err error) {
	if err = validateChat(messages, o); err != nil {
		return
	}
	if !m.info.HasChatTemplate {
		return result, fmt.Errorf("gofish: GGUF has no conversation template; set GGUFOptions.ChatTemplate to a Jinja template file")
	}
	if len(o.Tools) > 0 && (o.ToolMode == "json" || ((o.ToolMode == "" || o.ToolMode == "auto") && m.jsonToolFamily())) {
		return m.chatJSONTools(ctx, messages, o, options...)
	}
	if m.jsonToolFamily() {
		messages = normalizeToolHistory(messages)
	}
	c, err := generation.Resolve(options)
	if err != nil {
		return result, err
	}
	if len(o.ResponseSchema) > 0 && m.info.Architecture == "gemma" {
		return m.chatSchemaRaw(ctx, messages, o, c)
	}
	body := sampling(c)
	body["messages"] = messages
	body["stream"] = c.OnToken != nil || o.OnDelta != nil
	// Qwen3's embedded template receives this explicit non-thinking preference.
	body["chat_template_kwargs"] = map[string]any{"enable_thinking": false}
	if len(o.Tools) > 0 {
		body["tools"] = o.Tools
		choice := o.ToolChoice
		if choice == "" {
			choice = "auto"
		}
		body["tool_choice"] = choice
		body["parallel_tool_calls"] = false
	}
	if len(o.ResponseSchema) > 0 {
		body["response_format"] = map[string]any{"type": "json_schema", "json_schema": map[string]any{"name": "response", "strict": true, "schema": o.ResponseSchema}}
	}
	response, err := m.request(ctx, "/v1/chat/completions", body)
	if err != nil {
		return result, err
	}
	if !body["stream"].(bool) {
		var wire wireResponse
		if err = decode(response, &wire); err != nil {
			return result, err
		}
		if len(wire.Choices) != 1 {
			return result, fmt.Errorf("gofish: runtime returned no chat completion")
		}
		result = chat.Response{Message: wire.Choices[0].Message, FinishReason: wire.Choices[0].FinishReason, Usage: wire.Usage}
		return result, validateResult(result, o)
	}
	defer response.Body.Close()
	var content, reasoning strings.Builder
	defer func() {
		result.Message.Content = content.String()
		result.Message.ReasoningContent = reasoning.String()
	}()
	result.Message.Role = "assistant"
	err = readEvents(response.Body, func(data []byte) error {
		var wire wireResponse
		if e := json.Unmarshal(data, &wire); e != nil {
			return e
		}
		if wire.Error != nil && string(wire.Error) != "null" {
			return fmt.Errorf("gofish: runtime stream error: %s", wire.Error)
		}
		if wire.Usage.TotalTokens > 0 {
			result.Usage = wire.Usage
		}
		for _, choice := range wire.Choices {
			d := choice.Delta
			content.WriteString(d.Content)
			reasoning.WriteString(d.ReasoningContent)
			if choice.FinishReason != "" {
				result.FinishReason = choice.FinishReason
			}
			for _, t := range d.ToolCalls {
				if t.Index < 0 || t.Index > 128 {
					return fmt.Errorf("gofish: invalid streamed tool index")
				}
				for len(result.Message.ToolCalls) <= t.Index {
					result.Message.ToolCalls = append(result.Message.ToolCalls, chat.ToolCall{})
				}
				call := &result.Message.ToolCalls[t.Index]
				call.ID += t.ID
				if t.Type != "" {
					call.Type = t.Type
				}
				call.Function.Name += t.Function.Name
				call.Function.Arguments += t.Function.Arguments
			}
			if c.OnToken != nil && d.Content != "" {
				if e := c.OnToken(d.Content); e != nil {
					return e
				}
			}
			if o.OnDelta != nil {
				if e := o.OnDelta(d); e != nil {
					return e
				}
			}
		}
		return nil
	})
	if err == nil {
		result.Message.Content = content.String()
		result.Message.ReasoningContent = reasoning.String()
		err = validateResult(result, o)
	}
	return
}

// readEvents handles SSE event boundaries, multiline data and explicit completion.
// EOF without [DONE] is an interrupted response, not a successful generation.
func readEvents(r io.Reader, handle func([]byte) error) error {
	scanner := bufio.NewScanner(io.LimitReader(r, 64<<20))
	scanner.Buffer(make([]byte, 4096), 4<<20)
	var data []string
	dispatch := func() (bool, error) {
		if len(data) == 0 {
			return false, nil
		}
		payload := strings.Join(data, "\n")
		data = nil
		if payload == "[DONE]" {
			return true, nil
		}
		return false, handle([]byte(payload))
	}
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			done, err := dispatch()
			if err != nil {
				return err
			}
			if done {
				return nil
			}
			continue
		}
		if strings.HasPrefix(line, "data:") {
			data = append(data, strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " "))
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	done, err := dispatch()
	if err != nil {
		return err
	}
	if done {
		return nil
	}
	return fmt.Errorf("gofish: generation stream ended before [DONE]: %w", io.ErrUnexpectedEOF)
}

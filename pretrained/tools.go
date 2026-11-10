package pretrained

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/JWset/gofish/chat"
	"github.com/JWset/gofish/generation"
)

func (m *Model) jsonToolFamily() bool {
	return strings.HasPrefix(m.info.Family, "gemma") || strings.HasPrefix(m.info.Family, "smollm")
}

// normalizeToolHistory translates typed call/result messages for conversation
// templates without native tool roles. User-owned history is never mutated.
func normalizeToolHistory(messages []chat.Message) []chat.Message {
	out := append([]chat.Message(nil), messages...)
	for i := range out {
		if len(out[i].ToolCalls) > 0 {
			calls := make([]map[string]any, 0, len(out[i].ToolCalls))
			for _, call := range out[i].ToolCalls {
				calls = append(calls, map[string]any{"name": call.Function.Name, "arguments": json.RawMessage(call.Function.Arguments)})
			}
			data, _ := json.Marshal(map[string]any{"content": out[i].Content, "tool_calls": calls})
			out[i].Content = string(data)
			out[i].ToolCalls = nil
		}
		if out[i].Role == "tool" {
			out[i].Role = "user"
			out[i].Content = "Tool result for " + out[i].ToolCallID + ":\n" + out[i].Content
			out[i].ToolCallID = ""
			out[i].Name = ""
		}
	}
	return out
}

func (m *Model) chatJSONTools(ctx context.Context, messages []chat.Message, o chat.Options, options ...generation.Option) (result chat.Response, err error) {
	c, err := generation.Resolve(options)
	if err != nil {
		return result, err
	}
	alternatives := make([]any, 0, len(o.Tools))
	for _, tool := range o.Tools {
		alternatives = append(alternatives, map[string]any{"type": "object", "properties": map[string]any{"name": map[string]any{"const": tool.Function.Name}, "arguments": tool.Function.Parameters}, "required": []string{"name", "arguments"}, "additionalProperties": false})
	}
	minCalls, maxCalls := 0, 1
	if o.ToolChoice == "required" {
		minCalls = 1
	}
	if o.ToolChoice == "none" {
		maxCalls = 0
	}
	contentSchema := map[string]any{"type": "string"}
	if o.ToolChoice == "required" {
		contentSchema["maxLength"] = 0
	}
	schema, err := json.Marshal(map[string]any{"type": "object", "properties": map[string]any{"content": contentSchema, "tool_calls": map[string]any{"type": "array", "items": map[string]any{"oneOf": alternatives}, "minItems": minCalls, "maxItems": maxCalls}}, "required": []string{"content", "tool_calls"}, "additionalProperties": false})
	if err != nil {
		return result, err
	}
	definitions, _ := json.Marshal(o.Tools)
	instruction := "Available functions: " + string(definitions) + "\nReturn the required JSON object with content (a string) and tool_calls (an array). Each tool call has name and arguments. Use a function when needed; never invent results. Tool choice: " + o.ToolChoice + ".\n\n"
	messages = normalizeToolHistory(messages)
	found := false
	for i := range messages {
		if messages[i].Role == "user" {
			messages[i].Content = instruction + messages[i].Content
			found = true
			break
		}
	}
	if !found {
		return result, fmt.Errorf("gofish: JSON tool calling requires a user message")
	}
	// Buffer the schema envelope rather than exposing half-valid argument JSON.
	// Text streaming for ordinary Chat/Generate and native tool fragments remain live.
	inner := append([]generation.Option(nil), options...)
	inner = append(inner, generation.WithStream(nil))
	raw, err := m.ChatStructured(ctx, messages, chat.Options{ResponseSchema: schema}, inner...)
	if err != nil {
		return result, err
	}
	var envelope struct {
		Content string `json:"content"`
		Calls   []struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		} `json:"tool_calls"`
	}
	if err = json.Unmarshal([]byte(raw.Message.Content), &envelope); err != nil {
		return result, fmt.Errorf("gofish: invalid tool JSON: %w", err)
	}
	result = raw
	result.Message.Content = envelope.Content
	for _, call := range envelope.Calls {
		id := make([]byte, 12)
		if _, err = rand.Read(id); err != nil {
			return result, err
		}
		result.Message.ToolCalls = append(result.Message.ToolCalls, chat.ToolCall{ID: "call_" + hex.EncodeToString(id), Type: "function", Function: chat.FunctionCall{Name: call.Name, Arguments: string(call.Arguments)}})
	}
	if len(result.Message.ToolCalls) > 0 {
		result.FinishReason = "tool_calls"
	}
	if err = validateResult(result, o); err != nil {
		return result, err
	}
	if c.OnToken != nil && result.Message.Content != "" {
		if err = c.OnToken(result.Message.Content); err != nil {
			return result, err
		}
	}
	if o.OnDelta != nil {
		delta := chat.Delta{Content: result.Message.Content}
		for i, call := range result.Message.ToolCalls {
			delta.ToolCalls = append(delta.ToolCalls, chat.ToolDelta{Index: i, ID: call.ID, Type: call.Type, Function: call.Function})
		}
		if err = o.OnDelta(delta); err != nil {
			return result, err
		}
	}
	return result, nil
}

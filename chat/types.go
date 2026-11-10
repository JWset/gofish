// Package chat contains conversation and structured function-call types.
package chat

import "encoding/json"

type Message struct {
	Role             string     `json:"role"`
	Content          string     `json:"content"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string     `json:"tool_call_id,omitempty"`
	Name             string     `json:"name,omitempty"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
}
type Function struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
	Strict      bool            `json:"strict,omitempty"`
}
type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}
type ToolDelta struct {
	Index    int          `json:"index"`
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type,omitempty"`
	Function FunctionCall `json:"function"`
}
type Delta struct {
	Content          string      `json:"content,omitempty"`
	ReasoningContent string      `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolDelta `json:"tool_calls,omitempty"`
}
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}
type Response struct {
	Message      Message
	FinishReason string
	Usage        Usage
}
type Options struct {
	// ToolMode is auto (default), native, or json. JSON mode constrains calls with a schema.
	ToolMode   string
	Tools      []Tool
	ToolChoice string // "auto", "none", or "required". Empty defaults to auto.
	// ResponseSchema optionally constrains ordinary assistant text to a JSON schema.
	ResponseSchema json.RawMessage
	// OnDelta streams text, reasoning, and tool-call fragments. Returning an error cancels.
	OnDelta func(Delta) error
}

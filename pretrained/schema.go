package pretrained

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/JWset/gofish/chat"
	"github.com/JWset/gofish/generation"
)

// Gemma 1's legacy chat parser wraps schema grammars in special turn tokens
// that its grammar sampler cannot initialize. Apply its original template,
// then constrain the raw completion with the schema instead.
func (m *Model) chatSchemaRaw(ctx context.Context, messages []chat.Message, o chat.Options, c generation.Config) (result chat.Response, err error) {
	prompt, err := m.ApplyChatTemplate(ctx, messages)
	if err != nil {
		return result, err
	}
	body := sampling(c)
	body["prompt"] = prompt
	body["json_schema"] = o.ResponseSchema
	body["stream"] = c.OnToken != nil || o.OnDelta != nil
	response, err := m.request(ctx, "/v1/completions", body)
	if err != nil {
		return result, err
	}
	result.Message.Role = "assistant"
	if !body["stream"].(bool) {
		var wire wireResponse
		if err = decode(response, &wire); err != nil {
			return result, err
		}
		if len(wire.Choices) != 1 {
			return result, fmt.Errorf("gofish: runtime returned no schema completion")
		}
		result.Message.Content = wire.Choices[0].Text
		result.FinishReason = wire.Choices[0].FinishReason
		result.Usage = wire.Usage
		return result, validateResult(result, o)
	}
	defer response.Body.Close()
	var content strings.Builder
	err = readEvents(response.Body, func(data []byte) error {
		var wire wireResponse
		if err := json.Unmarshal(data, &wire); err != nil {
			return err
		}
		if wire.Error != nil && string(wire.Error) != "null" {
			return fmt.Errorf("gofish: runtime schema stream error: %s", wire.Error)
		}
		if wire.Usage.TotalTokens > 0 {
			result.Usage = wire.Usage
		}
		for _, choice := range wire.Choices {
			content.WriteString(choice.Text)
			if choice.FinishReason != "" {
				result.FinishReason = choice.FinishReason
			}
			if c.OnToken != nil && choice.Text != "" {
				if err := c.OnToken(choice.Text); err != nil {
					return err
				}
			}
			if o.OnDelta != nil {
				if err := o.OnDelta(chat.Delta{Content: choice.Text}); err != nil {
					return err
				}
			}
		}
		return nil
	})
	result.Message.Content = content.String()
	if err == nil {
		err = validateResult(result, o)
	}
	return
}

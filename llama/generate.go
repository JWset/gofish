package llama

import (
	"context"
	"fmt"
	"math/rand"
	"strings"
	"unicode/utf8"

	"github.com/JWset/gofish/generation"
	"github.com/JWset/gofish/tokenizer"
)

// Generate uses native quantized matrix-vector operations and a per-request KV
// cache. Prompts longer than the configured context keep their final tokens.
func (m *Model) Generate(ctx context.Context, prompt string, options ...generation.Option) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	config, err := generation.Resolve(options)
	if err != nil {
		return "", err
	}
	ids, err := m.vocab.encode(prompt)
	if err != nil {
		return "", err
	}
	if len(ids) == 0 {
		ids = []int{m.vocab.bos}
	}
	if len(ids) > m.config.ContextSize {
		ids = ids[len(ids)-m.config.ContextSize:]
	}
	session := m.newSession()
	var logits []float64
	for _, id := range ids {
		logits, err = session.step(ctx, id)
		if err != nil {
			return "", err
		}
	}
	rng := rand.New(rand.NewSource(config.Seed))
	collector := generation.NewCollector(config)
	pending := []byte{}
	first := true
	for step := 0; step < config.MaxTokens; step++ {
		if err = ctx.Err(); err != nil {
			return collector.Text(), err
		}
		selected, err := generation.Sample(logits, config, rng, func(id int) bool {
			if id == m.vocab.bos {
				return false
			}
			if id == m.vocab.eos {
				return len(pending) == 0
			}
			piece := m.vocab.piece(id)
			if piece == "" {
				return false
			}
			prefix := append([]byte(nil), pending...)
			for i := 0; i < len(piece); i++ {
				if !tokenizer.ValidNext(prefix, piece[i]) {
					return false
				}
				prefix = append(prefix, piece[i])
				if utf8.FullRune(prefix) {
					prefix = nil
				}
			}
			if len(prefix) == 0 {
				return true
			}
			width := 2
			if prefix[0] >= 0xf0 {
				width = 4
			} else if prefix[0] >= 0xe0 {
				width = 3
			}
			return width-len(prefix) <= config.MaxTokens-step-1
		})
		if err != nil {
			return collector.Text(), err
		}
		if selected == m.vocab.eos {
			break
		}
		piece := m.vocab.piece(selected)
		if first && prompt == "" && m.vocab.spacePrefix {
			piece = strings.TrimPrefix(piece, " ")
		}
		first = false
		pending = append(pending, []byte(piece)...)
		// Flush complete runes while retaining a partial final byte-fallback rune.
		n := 0
		for n < len(pending) && utf8.FullRune(pending[n:]) {
			_, size := utf8.DecodeRune(pending[n:])
			n += size
		}
		if n > 0 {
			stopped, err := collector.Feed(string(pending[:n]))
			pending = append([]byte(nil), pending[n:]...)
			if err != nil {
				return collector.Text(), err
			}
			if stopped {
				return collector.Text(), nil
			}
		}
		if step+1 < config.MaxTokens {
			logits, err = session.step(ctx, selected)
			if err != nil {
				return collector.Text(), err
			}
		}
	}
	err = collector.Finish()
	return collector.Text(), err
}

// Encode exposes the model's tokenizer, including configured BOS/EOS insertion.
func (m *Model) Encode(text string) ([]int, error) { return m.vocab.encode(text) }

// Decode skips control tokens and removes SentencePiece's one dummy prefix space.
func (m *Model) Decode(ids []int) (string, error) {
	var builder strings.Builder
	for _, id := range ids {
		if id < 0 || id >= len(m.vocab.pieces) {
			return "", fmt.Errorf("gofish: token ID %d out of range", id)
		}
		builder.WriteString(m.vocab.piece(id))
	}
	text := builder.String()
	if m.vocab.spacePrefix {
		text = strings.TrimPrefix(text, " ")
	}
	if !utf8.ValidString(text) {
		return "", fmt.Errorf("gofish: incomplete or invalid UTF-8 token sequence")
	}
	return text, nil
}

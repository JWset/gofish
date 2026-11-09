// Package tokenizer provides a lossless byte tokenizer for native transformers.
package tokenizer

import (
	"fmt"
	"unicode/utf8"
)

const BOS = 256
const VocabularySize = 257

// Byte maps each UTF-8 byte to its integer value. BOS is a separate context token.
// No learned vocabulary or external tokenizer files are required.
type Byte struct{}

func (Byte) Encode(text string) []int {
	out := make([]int, len(text))
	for i := 0; i < len(text); i++ {
		out[i] = int(text[i])
	}
	return out
}
func (Byte) Decode(ids []int) (string, error) {
	data := make([]byte, len(ids))
	for i, id := range ids {
		if id < 0 || id > 255 {
			return "", fmt.Errorf("gofish: token %d is not a byte", id)
		}
		data[i] = byte(id)
	}
	if !utf8.Valid(data) {
		return "", fmt.Errorf("gofish: incomplete or invalid UTF-8 token sequence")
	}
	return string(data), nil
}

// ValidNext reports whether next can extend a complete UTF-8 string's unfinished
// final codepoint. pending must contain only the bytes of that unfinished codepoint.
func ValidNext(pending []byte, next byte) bool {
	candidate := append(append([]byte(nil), pending...), next)
	r, n := utf8.DecodeRune(candidate)
	return !(r == utf8.RuneError && n == 1 && utf8.FullRune(candidate))
}

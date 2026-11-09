package llama

import (
	"container/heap"
	"fmt"
	"math"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/JWset/gofish/gguf"
)

type vocabulary struct {
	model                       string
	pieces                      []string
	scores                      []float32
	types                       []int32
	lookup                      map[string]int
	bytes                       [256]int
	bos, eos                    int
	addBOS, addEOS, spacePrefix bool
}

func readVocabulary(f *gguf.File) (*vocabulary, error) {
	model := f.String("tokenizer.ggml.model", "")
	if model != "llama" && model != "gofish-byte" {
		return nil, fmt.Errorf("gofish: unsupported GGUF tokenizer %q; supported: llama SentencePiece and gofish-byte", model)
	}
	pieces, ok := f.Metadata["tokenizer.ggml.tokens"].([]string)
	if !ok || len(pieces) < 2 || len(pieces) > 262144 {
		return nil, fmt.Errorf("gofish: invalid GGUF tokenizer vocabulary")
	}
	v := &vocabulary{model: model, pieces: pieces, lookup: map[string]int{}, bos: int(f.Uint("tokenizer.ggml.bos_token_id", 1)), eos: int(f.Uint("tokenizer.ggml.eos_token_id", 2)), addBOS: f.Bool("tokenizer.ggml.add_bos_token", true), addEOS: f.Bool("tokenizer.ggml.add_eos_token", false), spacePrefix: f.Bool("tokenizer.ggml.add_space_prefix", true)}
	if v.bos < 0 || v.bos >= len(pieces) || v.eos < 0 || v.eos >= len(pieces) {
		return nil, fmt.Errorf("gofish: invalid special-token IDs")
	}
	v.scores, _ = f.Metadata["tokenizer.ggml.scores"].([]float32)
	v.types, _ = f.Metadata["tokenizer.ggml.token_type"].([]int32)
	if model == "llama" && (len(v.scores) != len(pieces) || len(v.types) != len(pieces)) {
		return nil, fmt.Errorf("gofish: SentencePiece requires scores and token types for every token")
	}
	if model == "gofish-byte" && (len(pieces) != 257 || v.bos != 256) {
		return nil, fmt.Errorf("gofish: gofish-byte requires 257 tokens and BOS 256")
	}
	for i := range v.bytes {
		v.bytes[i] = -1
	}
	for i, piece := range pieces {
		if _, exists := v.lookup[piece]; exists {
			return nil, fmt.Errorf("gofish: duplicate vocabulary piece")
		}
		v.lookup[piece] = i
		if model == "gofish-byte" && i < 256 {
			v.bytes[i] = i
			continue
		}
		if len(v.scores) > i && (math.IsNaN(float64(v.scores[i])) || math.IsInf(float64(v.scores[i]), 0)) {
			return nil, fmt.Errorf("gofish: non-finite tokenizer score")
		}
		if len(v.types) > i && v.types[i] == 6 {
			if len(piece) != 6 || !strings.HasPrefix(piece, "<0x") || piece[5] != '>' {
				return nil, fmt.Errorf("gofish: invalid byte fallback token")
			}
			b, err := strconv.ParseUint(piece[3:5], 16, 8)
			if err != nil {
				return nil, err
			}
			v.bytes[b] = i
		}
	}
	for _, id := range v.bytes {
		if id < 0 {
			return nil, fmt.Errorf("gofish: tokenizer must provide all 256 byte fallbacks")
		}
	}
	if f.Bool("tokenizer.ggml.remove_extra_whitespaces", false) {
		return nil, fmt.Errorf("gofish: whitespace-normalizing tokenizers are not supported")
	}
	return v, nil
}

type pair struct {
	left, right, size int
	score             float32
}
type pairs []pair

func (p pairs) Len() int { return len(p) }
func (p pairs) Less(i, j int) bool {
	if p[i].score == p[j].score {
		return p[i].left < p[j].left
	}
	return p[i].score > p[j].score
}
func (p pairs) Swap(i, j int) { p[i], p[j] = p[j], p[i] }
func (p *pairs) Push(x any)   { *p = append(*p, x.(pair)) }
func (p *pairs) Pop() any     { last := (*p)[len(*p)-1]; *p = (*p)[:len(*p)-1]; return last }

type symbol struct{ start, end, previous, next int }

func (v *vocabulary) encode(text string) ([]int, error) {
	if !utf8.ValidString(text) {
		return nil, fmt.Errorf("gofish: prompt must be valid UTF-8")
	}
	ids := []int{}
	if v.addBOS {
		ids = append(ids, v.bos)
	}
	if v.model == "gofish-byte" {
		for i := 0; i < len(text); i++ {
			ids = append(ids, int(text[i]))
		}
		return ids, nil
	}
	if text != "" {
		if v.spacePrefix {
			text = " " + text
		}
		text = strings.ReplaceAll(text, " ", "▁")
		symbols := []symbol{}
		for start, r := range text {
			end := start + utf8.RuneLen(r)
			symbols = append(symbols, symbol{start: start, end: end, previous: len(symbols) - 1, next: len(symbols) + 1})
		}
		symbols[len(symbols)-1].next = -1
		queue := &pairs{}
		heap.Init(queue)
		add := func(left, right int) {
			if left < 0 || right < 0 {
				return
			}
			piece := text[symbols[left].start:symbols[right].end]
			if id, ok := v.lookup[piece]; ok && v.types[id] != 3 && v.types[id] != 5 {
				heap.Push(queue, pair{left: left, right: right, size: len(piece), score: v.scores[id]})
			}
		}
		for i := 1; i < len(symbols); i++ {
			add(i-1, i)
		}
		for queue.Len() > 0 {
			candidate := heap.Pop(queue).(pair)
			left, right := &symbols[candidate.left], &symbols[candidate.right]
			if left.end == left.start || right.end == right.start || left.next != candidate.right || right.end-left.start != candidate.size {
				continue
			}
			left.end = right.end
			right.start = right.end
			left.next = right.next
			if right.next >= 0 {
				symbols[right.next].previous = candidate.left
			}
			add(left.previous, candidate.left)
			add(candidate.left, left.next)
		}
		for i := 0; i >= 0; i = symbols[i].next {
			piece := text[symbols[i].start:symbols[i].end]
			if id, ok := v.lookup[piece]; ok && v.types[id] != 3 && v.types[id] != 5 {
				ids = append(ids, id)
			} else {
				for j := 0; j < len(piece); j++ {
					ids = append(ids, v.bytes[piece[j]])
				}
			}
		}
	}
	if v.addEOS {
		ids = append(ids, v.eos)
	}
	return ids, nil
}
func (v *vocabulary) piece(id int) string {
	if id < 0 || id >= len(v.pieces) || id == v.bos || id == v.eos {
		return ""
	}
	if v.model == "gofish-byte" {
		return string([]byte{byte(id)})
	}
	if v.types[id] == 3 || v.types[id] == 5 || v.types[id] == 2 {
		return ""
	}
	if v.types[id] == 6 {
		for b, token := range v.bytes {
			if token == id {
				return string([]byte{byte(b)})
			}
		}
	}
	return strings.ReplaceAll(v.pieces[id], "▁", " ")
}

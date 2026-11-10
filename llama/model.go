// Package llama runs a supported subset of GGUF Llama models directly in Go.
// It implements RMSNorm, interleaved RoPE, grouped-query attention with a bounded
// KV cache, and SwiGLU. No external inference server, CGO or Python is required.
package llama

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/JWset/gofish/gguf"
)

type Config struct {
	ContextSize      int
	Width            int
	Heads            int
	KVHeads          int
	Layers           int
	FeedForward      int
	RotaryDimensions int
	RopeBase         float64
	RopeScale        float64
	Epsilon          float64
	Seed             int64
}
type LoadOptions struct {
	// Runtime selects auto (default), native, or llama.cpp at the root GoFish API.
	Runtime        string
	ServerPath     string
	GPULayers      int
	Threads        int
	ChatTemplate   string
	StartupTimeout time.Duration
	ContextSize    int    // Zero uses min(declared context, 2048).
	MaxModelBytes  uint64 // Zero: native 2 GiB tensor budget; managed runtime 16 GiB file budget.
	MaxCacheBytes  uint64 // Zero: native 256 MiB/session; managed runtime 1 GiB estimated KV budget.
}
type layer struct {
	norm, ffnNorm                   []float32
	q, k, v, output, gate, up, down *gguf.Matrix
}
type Model struct {
	mu                sync.Mutex
	config            Config
	vocab             *vocabulary
	embedding, output *gguf.Matrix
	finalNorm         []float32
	layers            []layer
	matrices          map[string]*gguf.Matrix
	norms             map[string][]float32
	parameterCount    int
	maxCacheBytes     uint64
	version           uint64
	metadata          map[string]any
}

func Load(path string) (*Model, error) { return LoadWithOptions(path, LoadOptions{}) }
func LoadWithOptions(path string, options LoadOptions) (*Model, error) {
	f, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return fromGGUF(f, options)
}
func (m *Model) Config() Config      { return m.config }
func (m *Model) ParameterCount() int { return m.parameterCount }
func (m *Model) CanTrain() bool      { return m.parameterCount <= 2000000 }

func fromGGUF(f *gguf.File, options LoadOptions) (*Model, error) {
	if f.Uint("split.count", 1) > 1 {
		return nil, fmt.Errorf("gofish: sharded GGUF models are unsupported")
	}
	if f.String("general.architecture", "") != "llama" {
		return nil, fmt.Errorf("gofish: native GGUF supports architecture llama, got %q", f.String("general.architecture", ""))
	}
	if f.Uint("llama.expert_count", 0) > 0 || f.Uint("llama.attention.sliding_window", 0) > 0 || !f.Bool("llama.attention.causal", true) {
		return nil, fmt.Errorf("gofish: MoE, declared sliding-window or noncausal Llama variants are unsupported")
	}
	vocab, err := readVocabulary(f)
	if err != nil {
		return nil, err
	}
	c := Config{ContextSize: int(f.Uint("llama.context_length", 0)), Width: int(f.Uint("llama.embedding_length", 0)), Heads: int(f.Uint("llama.attention.head_count", 0)), Layers: int(f.Uint("llama.block_count", 0)), FeedForward: int(f.Uint("llama.feed_forward_length", 0)), RopeBase: f.Float("llama.rope.freq_base", 10000), RopeScale: 1, Epsilon: f.Float("llama.attention.layer_norm_rms_epsilon", 1e-5)}
	c.KVHeads = int(f.Uint("llama.attention.head_count_kv", uint64(c.Heads)))
	if c.Width < 1 || c.Width > 16384 || c.Heads < 1 || c.Heads > 256 || c.Width%c.Heads != 0 || c.KVHeads < 1 || c.KVHeads > c.Heads || c.Heads%c.KVHeads != 0 || c.Layers < 1 || c.Layers > 128 || c.FeedForward < 1 || c.FeedForward > 131072 || c.ContextSize < 1 || c.ContextSize > 1048576 {
		return nil, fmt.Errorf("gofish: invalid or excessive Llama dimensions")
	}
	headWidth := c.Width / c.Heads
	c.RotaryDimensions = int(f.Uint("llama.rope.dimension_count", uint64(headWidth)))
	if c.RotaryDimensions < 2 || c.RotaryDimensions > headWidth || c.RotaryDimensions%2 != 0 || !positive(c.RopeBase) || !positive(c.Epsilon) {
		return nil, fmt.Errorf("gofish: invalid Llama normalization or rotary configuration")
	}
	if f.Uint("llama.attention.key_length", uint64(headWidth)) != uint64(headWidth) || f.Uint("llama.attention.value_length", uint64(headWidth)) != uint64(headWidth) {
		return nil, fmt.Errorf("gofish: unequal attention head dimensions are unsupported")
	}
	scaling := f.String("llama.rope.scaling.type", "none")
	if scaling == "linear" {
		factor := f.Float("llama.rope.scaling.factor", 0)
		if !positive(factor) {
			return nil, fmt.Errorf("gofish: invalid linear RoPE factor")
		}
		c.RopeScale = 1 / factor
	} else if scaling != "none" {
		return nil, fmt.Errorf("gofish: unsupported RoPE scaling %q", scaling)
	}
	if options.ContextSize < 0 || options.ContextSize > c.ContextSize {
		return nil, fmt.Errorf("gofish: requested context exceeds model context")
	}
	if options.ContextSize == 0 {
		c.ContextSize = min(c.ContextSize, 2048)
	} else {
		c.ContextSize = options.ContextSize
	}
	if options.MaxModelBytes == 0 {
		options.MaxModelBytes = 2 << 30
	}
	if options.MaxCacheBytes == 0 {
		options.MaxCacheBytes = 256 << 20
	}
	cacheBytes := uint64(c.Layers) * uint64(c.ContextSize) * uint64(c.KVHeads) * uint64(headWidth) * 8
	if cacheBytes > options.MaxCacheBytes {
		return nil, fmt.Errorf("gofish: KV cache needs %d bytes, exceeding budget; reduce context", cacheBytes)
	}
	type expected struct {
		name       string
		dimensions []uint64
	}
	width, vocabulary, ff, kv := uint64(c.Width), uint64(len(vocab.pieces)), uint64(c.FeedForward), uint64(c.KVHeads*headWidth)
	list := []expected{{"token_embd.weight", []uint64{width, vocabulary}}, {"output_norm.weight", []uint64{width}}}
	if _, ok := f.Tensor("output.weight"); ok {
		list = append(list, expected{"output.weight", []uint64{width, vocabulary}})
	}
	for i := 0; i < c.Layers; i++ {
		prefix := fmt.Sprintf("blk.%d.", i)
		for _, e := range []expected{{"attn_norm.weight", []uint64{width}}, {"attn_q.weight", []uint64{width, width}}, {"attn_k.weight", []uint64{width, kv}}, {"attn_v.weight", []uint64{width, kv}}, {"attn_output.weight", []uint64{width, width}}, {"ffn_norm.weight", []uint64{width}}, {"ffn_gate.weight", []uint64{width, ff}}, {"ffn_up.weight", []uint64{width, ff}}, {"ffn_down.weight", []uint64{ff, width}}} {
			e.name = prefix + e.name
			list = append(list, e)
		}
	}
	required := map[string]bool{}
	total := uint64(0)
	count := uint64(0)
	for _, e := range list {
		t, ok := f.Tensor(e.name)
		if !ok {
			return nil, fmt.Errorf("gofish: missing Llama tensor %q", e.name)
		}
		if len(t.Dimensions) != len(e.dimensions) {
			return nil, fmt.Errorf("gofish: wrong rank for %q", e.name)
		}
		for i, d := range e.dimensions {
			if t.Dimensions[i] != d {
				return nil, fmt.Errorf("gofish: wrong shape for %q", e.name)
			}
		}
		if t.Size == 0 || t.Size > options.MaxModelBytes-total {
			return nil, fmt.Errorf("gofish: unsupported encoding or model byte budget exceeded at %q (%s)", e.name, t.Type)
		}
		total += t.Size
		count += t.Elements()
		required[e.name] = true
	}
	for _, t := range f.Tensors {
		if !required[t.Name] {
			return nil, fmt.Errorf("gofish: unsupported extra Llama tensor %q", t.Name)
		}
	}
	if count > uint64(int(^uint(0)>>1)) {
		return nil, fmt.Errorf("gofish: parameter count exceeds platform integer range")
	}
	m := &Model{config: c, vocab: vocab, matrices: map[string]*gguf.Matrix{}, norms: map[string][]float32{}, parameterCount: int(count), maxCacheBytes: options.MaxCacheBytes}
	m.metadata = make(map[string]any, len(f.Metadata))
	for key, value := range f.Metadata {
		m.metadata[key] = value
	}
	for _, e := range list {
		t, _ := f.Tensor(e.name)
		if len(t.Dimensions) == 1 {
			values, err := t.Decode(width)
			if err != nil {
				return nil, err
			}
			for _, value := range values {
				if !finite(float64(value)) {
					return nil, fmt.Errorf("gofish: non-finite normalization weight")
				}
			}
			m.norms[e.name] = values
		} else {
			matrix, err := gguf.NewMatrix(t, t.Size)
			if err != nil {
				return nil, err
			}
			m.matrices[e.name] = matrix
		}
	}
	m.bind()
	return m, nil
}
func (m *Model) bind() {
	m.embedding = m.matrices["token_embd.weight"]
	m.output = m.matrices["output.weight"]
	if m.output == nil {
		m.output = m.embedding
	}
	m.finalNorm = m.norms["output_norm.weight"]
	m.layers = nil
	for i := 0; i < m.config.Layers; i++ {
		p := fmt.Sprintf("blk.%d.", i)
		m.layers = append(m.layers, layer{norm: m.norms[p+"attn_norm.weight"], ffnNorm: m.norms[p+"ffn_norm.weight"], q: m.matrices[p+"attn_q.weight"], k: m.matrices[p+"attn_k.weight"], v: m.matrices[p+"attn_v.weight"], output: m.matrices[p+"attn_output.weight"], gate: m.matrices[p+"ffn_gate.weight"], up: m.matrices[p+"ffn_up.weight"], down: m.matrices[p+"ffn_down.weight"]})
	}
}
func finite(v float64) bool   { return !math.IsNaN(v) && !math.IsInf(v, 0) }
func positive(v float64) bool { return finite(v) && v > 0 }

type cache struct{ keys, values []float32 }

// Session owns a bounded KV cache. Step is incremental, and preserves absolute
// rotary positions when old context is evicted. A session is not goroutine safe.
type Session struct {
	model    *Model
	position int
	version  uint64
	layers   []cache
}

func (m *Model) newSession() *Session {
	s := &Session{model: m, version: m.version}
	n := m.config.ContextSize * m.config.KVHeads * (m.config.Width / m.config.Heads)
	for i := 0; i < m.config.Layers; i++ {
		s.layers = append(s.layers, cache{keys: make([]float32, n), values: make([]float32, n)})
	}
	return s
}
func (m *Model) NewSession() *Session { m.mu.Lock(); defer m.mu.Unlock(); return m.newSession() }
func (s *Session) Reset() {
	s.model.mu.Lock()
	defer s.model.mu.Unlock()
	s.position = 0
	s.version = s.model.version
	for _, c := range s.layers {
		clear(c.keys)
		clear(c.values)
	}
}
func (s *Session) Step(ctx context.Context, token int) ([]float64, error) {
	s.model.mu.Lock()
	defer s.model.mu.Unlock()
	return s.step(ctx, token)
}
func rms(x, weight []float32, epsilon float64) []float32 {
	sum := 0.0
	for _, v := range x {
		sum += float64(v) * float64(v)
	}
	scale := 1 / math.Sqrt(sum/float64(len(x))+epsilon)
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = float32(float64(v) * scale * float64(weight[i]))
	}
	return out
}
func rotary(x []float32, heads, dimensions, position int, base, scale float64) {
	width := len(x) / heads
	for head := 0; head < heads; head++ {
		for j := 0; j < dimensions; j += 2 {
			angle := float64(position) * scale * math.Pow(base, -float64(j)/float64(dimensions))
			s, c := math.Sincos(angle)
			index := head*width + j
			a, b := float64(x[index]), float64(x[index+1])
			x[index] = float32(a*c - b*s)
			x[index+1] = float32(a*s + b*c)
		}
	}
}
func add(x, y []float32) {
	for i := range x {
		x[i] += y[i]
	}
}
func (s *Session) step(ctx context.Context, token int) ([]float64, error) {
	m := s.model
	c := m.config
	if s.version != m.version {
		return nil, fmt.Errorf("gofish: model changed; reset the KV session")
	}
	if token < 0 || token >= len(m.vocab.pieces) {
		return nil, fmt.Errorf("gofish: token out of range")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	x := make([]float32, c.Width)
	if err := m.embedding.Row(token, x); err != nil {
		return nil, err
	}
	headWidth := c.Width / c.Heads
	kvWidth := c.KVHeads * headWidth
	for i, l := range m.layers {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		normalized := rms(x, l.norm, c.Epsilon)
		q, _ := l.q.MatVec(normalized)
		k, _ := l.k.MatVec(normalized)
		v, _ := l.v.MatVec(normalized)
		rotary(q, c.Heads, c.RotaryDimensions, s.position, c.RopeBase, c.RopeScale)
		rotary(k, c.KVHeads, c.RotaryDimensions, s.position, c.RopeBase, c.RopeScale)
		state := s.layers[i]
		slot := s.position % c.ContextSize
		copy(state.keys[slot*kvWidth:], k)
		copy(state.values[slot*kvWidth:], v)
		attention := make([]float32, c.Width)
		start := max(0, s.position-c.ContextSize+1)
		scores := make([]float64, s.position-start+1)
		for head := 0; head < c.Heads; head++ {
			kvHead := head / (c.Heads / c.KVHeads)
			peak := math.Inf(-1)
			for position := start; position <= s.position; position++ {
				offset := (position%c.ContextSize)*kvWidth + kvHead*headWidth
				score := 0.0
				for j := 0; j < headWidth; j++ {
					score += float64(q[head*headWidth+j]) * float64(state.keys[offset+j])
				}
				score /= math.Sqrt(float64(headWidth))
				scores[position-start] = score
				peak = max(peak, score)
			}
			sum := 0.0
			for j, score := range scores {
				scores[j] = math.Exp(score - peak)
				sum += scores[j]
			}
			for j := 0; j < headWidth; j++ {
				value := 0.0
				for position := start; position <= s.position; position++ {
					offset := (position%c.ContextSize)*kvWidth + kvHead*headWidth + j
					value += (scores[position-start] / sum) * float64(state.values[offset])
				}
				attention[head*headWidth+j] = float32(value)
			}
		}
		projected, _ := l.output.MatVec(attention)
		add(x, projected)
		normalized = rms(x, l.ffnNorm, c.Epsilon)
		gate, _ := l.gate.MatVec(normalized)
		up, _ := l.up.MatVec(normalized)
		for j, g := range gate {
			value := float64(g)
			sigmoid := 0.0
			if value >= 0 {
				sigmoid = 1 / (1 + math.Exp(-value))
			} else {
				e := math.Exp(value)
				sigmoid = e / (1 + e)
			}
			gate[j] = float32(value * sigmoid * float64(up[j]))
		}
		down, _ := l.down.MatVec(gate)
		add(x, down)
	}
	normalized := rms(x, m.finalNorm, c.Epsilon)
	logits, _ := m.output.MatVec(normalized)
	out := make([]float64, len(logits))
	for i, v := range logits {
		out[i] = float64(v)
		if !finite(out[i]) {
			return nil, fmt.Errorf("gofish: non-finite GGUF logits")
		}
	}
	s.position++
	return out, nil
}

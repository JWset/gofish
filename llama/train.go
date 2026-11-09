package llama

import (
	"bytes"
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"

	"github.com/JWset/gofish/gguf"
	"github.com/JWset/gofish/optim"
	"github.com/JWset/gofish/tensor"
	"github.com/JWset/gofish/training"
)

// NewByteModel creates a small Llama-shaped model with a GoFish byte tokenizer.
// Its exported GGUF uses architecture llama, with custom tokenizer gofish-byte.
// Third-party runtimes need explicit support for that custom tokenizer.
func NewByteModel(config Config) (*Model, error) {
	c := config
	if c.ContextSize == 0 {
		c.ContextSize = 32
	}
	if c.Width == 0 {
		c.Width = 32
	}
	if c.Heads == 0 {
		c.Heads = 4
	}
	if c.KVHeads == 0 {
		c.KVHeads = c.Heads
	}
	if c.Layers == 0 {
		c.Layers = 1
	}
	if c.FeedForward == 0 {
		c.FeedForward = 2 * c.Width
	}
	if c.Epsilon == 0 {
		c.Epsilon = 1e-5
	}
	if c.RopeBase == 0 {
		c.RopeBase = 10000
	}
	if c.Width < 1 || c.Width > 512 || c.Heads < 1 || c.Width%c.Heads != 0 || c.KVHeads < 1 || c.Heads%c.KVHeads != 0 || c.Layers < 1 || c.Layers > 8 || c.FeedForward < 1 || c.FeedForward > 2048 || c.ContextSize < 1 || c.ContextSize > 256 || !positive(c.Epsilon) || !positive(c.RopeBase) {
		return nil, fmt.Errorf("gofish: invalid small Llama configuration")
	}
	if c.RotaryDimensions == 0 {
		c.RotaryDimensions = c.Width / c.Heads
	}
	if c.RotaryDimensions < 2 || c.RotaryDimensions%2 != 0 || c.RotaryDimensions > c.Width/c.Heads {
		return nil, fmt.Errorf("gofish: invalid rotary dimensions")
	}
	if c.RopeScale != 0 && c.RopeScale != 1 {
		return nil, fmt.Errorf("gofish: NewByteModel supports unscaled rotary embeddings")
	}
	kv := c.KVHeads * (c.Width / c.Heads)
	count := 257*c.Width*2 + c.Width + c.Layers*(2*c.Width+2*c.Width*c.Width+2*c.Width*kv+3*c.Width*c.FeedForward)
	if count > 2000000 {
		return nil, fmt.Errorf("gofish: new Llama exceeds 2 million parameter budget")
	}
	pieces := make([]string, 257)
	types := make([]int32, 257)
	scores := make([]float32, 257)
	for i := 0; i < 256; i++ {
		pieces[i] = fmt.Sprintf("<0x%02X>", i)
		types[i] = 6
	}
	pieces[256] = "<s>"
	types[256] = 3
	metadata := map[string]any{"general.architecture": "llama", "general.name": "GoFish byte Llama", "general.license": "CC-BY-SA-3.0", "llama.context_length": uint32(c.ContextSize), "llama.embedding_length": uint32(c.Width), "llama.block_count": uint32(c.Layers), "llama.feed_forward_length": uint32(c.FeedForward), "llama.attention.head_count": uint32(c.Heads), "llama.attention.head_count_kv": uint32(c.KVHeads), "llama.attention.layer_norm_rms_epsilon": float32(c.Epsilon), "llama.rope.freq_base": float32(c.RopeBase), "llama.rope.dimension_count": uint32(c.RotaryDimensions), "tokenizer.ggml.model": "gofish-byte", "tokenizer.ggml.tokens": pieces, "tokenizer.ggml.scores": scores, "tokenizer.ggml.token_type": types, "tokenizer.ggml.bos_token_id": uint32(256), "tokenizer.ggml.eos_token_id": uint32(256), "tokenizer.ggml.add_bos_token": true, "tokenizer.ggml.add_eos_token": false, "tokenizer.ggml.add_space_prefix": false}
	rng := rand.New(rand.NewSource(c.Seed))
	var tensors []gguf.TensorData
	matrix := func(name string, rows, columns int) {
		values := make([]float32, rows*columns)
		for i := range values {
			values[i] = float32(rng.NormFloat64() * 0.02)
		}
		tensors = append(tensors, gguf.TensorData{Name: name, Dimensions: []uint64{uint64(columns), uint64(rows)}, Type: gguf.F32, Data: gguf.Float32Data(values)})
	}
	norm := func(name string) {
		values := make([]float32, c.Width)
		for i := range values {
			values[i] = 1
		}
		tensors = append(tensors, gguf.TensorData{Name: name, Dimensions: []uint64{uint64(c.Width)}, Type: gguf.F32, Data: gguf.Float32Data(values)})
	}
	matrix("token_embd.weight", 257, c.Width)
	matrix("output.weight", 257, c.Width)
	norm("output_norm.weight")
	for i := 0; i < c.Layers; i++ {
		p := fmt.Sprintf("blk.%d.", i)
		norm(p + "attn_norm.weight")
		norm(p + "ffn_norm.weight")
		matrix(p+"attn_q.weight", c.Width, c.Width)
		matrix(p+"attn_k.weight", kv, c.Width)
		matrix(p+"attn_v.weight", kv, c.Width)
		matrix(p+"attn_output.weight", c.Width, c.Width)
		matrix(p+"ffn_gate.weight", c.FeedForward, c.Width)
		matrix(p+"ffn_up.weight", c.FeedForward, c.Width)
		matrix(p+"ffn_down.weight", c.Width, c.FeedForward)
	}
	var buffer bytes.Buffer
	if err := gguf.Write(&buffer, metadata, tensors); err != nil {
		return nil, err
	}
	f, err := gguf.Read(bytes.NewReader(buffer.Bytes()), int64(buffer.Len()))
	if err != nil {
		return nil, err
	}
	return fromGGUF(f, LoadOptions{ContextSize: c.ContextSize})
}
func keys[T any](values map[string]T) []string {
	out := make([]string, 0, len(values))
	for key := range values {
		out = append(out, key)
	}
	sort.Strings(out)
	return out
}
func (m *Model) parameters(track bool) (map[string]*tensor.Tensor, []*tensor.Tensor, error) {
	if !m.CanTrain() {
		return nil, nil, fmt.Errorf("gofish: GGUF training is limited to 2 million parameters; this model has %d", m.parameterCount)
	}
	weights := map[string]*tensor.Tensor{}
	var parameters []*tensor.Tensor
	for _, name := range keys(m.matrices) {
		matrix := m.matrices[name]
		values, err := matrix.Values(2000000)
		if err != nil {
			return nil, nil, err
		}
		data := make([]float64, len(values))
		for i, v := range values {
			data[i] = float64(v)
		}
		t := tensor.New(data, matrix.Rows, matrix.Columns)
		if track {
			t.WithGrad()
		}
		weights[name] = t
		parameters = append(parameters, t)
	}
	for _, name := range keys(m.norms) {
		data := make([]float64, len(m.norms[name]))
		for i, v := range m.norms[name] {
			data[i] = float64(v)
		}
		t := tensor.New(data, len(data))
		if track {
			t.WithGrad()
		}
		weights[name] = t
		parameters = append(parameters, t)
	}
	return weights, parameters, nil
}
func (m *Model) forward(ids []int, batch int, p map[string]*tensor.Tensor) *tensor.Tensor {
	c := m.config
	time := len(ids) / batch
	headWidth := c.Width / c.Heads
	linear := func(x *tensor.Tensor, name string) *tensor.Tensor {
		w := p[name]
		if w == nil && name == "output.weight" {
			w = p["token_embd.weight"]
		}
		return x.MatMul(w.Transpose())
	}
	x := p["token_embd.weight"].Gather(ids).Reshape(batch, time, c.Width)
	for i := 0; i < c.Layers; i++ {
		prefix := fmt.Sprintf("blk.%d.", i)
		norm := x.RMSNorm(c.Epsilon).Mul(p[prefix+"attn_norm.weight"])
		q := linear(norm, prefix+"attn_q.weight").RoPE(c.Heads, c.RotaryDimensions, 0, c.RopeBase, c.RopeScale)
		k := linear(norm, prefix+"attn_k.weight").RoPE(c.KVHeads, c.RotaryDimensions, 0, c.RopeBase, c.RopeScale).RepeatHeads(headWidth, c.Heads/c.KVHeads)
		v := linear(norm, prefix+"attn_v.weight").RepeatHeads(headWidth, c.Heads/c.KVHeads)
		x = x.Add(linear(tensor.Attention(q, k, v, c.Heads, true), prefix+"attn_output.weight"))
		norm = x.RMSNorm(c.Epsilon).Mul(p[prefix+"ffn_norm.weight"])
		x = x.Add(linear(linear(norm, prefix+"ffn_gate.weight").SiLU().Mul(linear(norm, prefix+"ffn_up.weight")), prefix+"ffn_down.weight"))
	}
	return linear(x.RMSNorm(c.Epsilon).Mul(p["output_norm.weight"]), "output.weight")
}
func (m *Model) syncWeights(p map[string]*tensor.Tensor) error {
	matrices := map[string]*gguf.Matrix{}
	norms := map[string][]float32{}
	for name, weight := range p {
		data := weight.Data()
		values := make([]float32, len(data))
		for i, value := range data {
			values[i] = float32(value)
			if !finite(float64(values[i])) {
				return fmt.Errorf("gofish: updated weights overflow float32")
			}
		}
		if old, ok := m.matrices[name]; ok {
			matrix, err := gguf.NewPackedMatrix(old.Rows, old.Columns, gguf.F32, gguf.Float32Data(values))
			if err != nil {
				return err
			}
			matrices[name] = matrix
		} else {
			norms[name] = values
		}
	}
	m.matrices = matrices
	m.norms = norms
	m.bind()
	m.version++
	return nil
}

// Train fine-tunes small GGUF models by dequantizing weights into the autograd
// backend. It starts fresh AdamW state. Saving exports float32 weights, without
// optimizer moments; Resume is unsupported for this GGUF path.
func (m *Model) Train(ctx context.Context, corpus string, options training.Options) (history training.History, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if err = ctx.Err(); err != nil {
		return
	}
	if options.Resume {
		return history, fmt.Errorf("gofish: GGUF does not contain resumable optimizer state")
	}
	ids, err := m.vocab.encode(corpus)
	if err != nil {
		return
	}
	if len(ids) < 3 {
		return history, fmt.Errorf("gofish: training text is too short")
	}
	if options.Steps == 0 {
		options.Steps = 500
	}
	if options.BatchSize == 0 {
		options.BatchSize = 4
	}
	if options.LearningRate == 0 {
		options.LearningRate = 0.003
	}
	if options.ClipNorm == 0 {
		options.ClipNorm = 1
	}
	if options.Steps < 1 || options.BatchSize < 1 || options.BatchSize > 256 || !positive(options.LearningRate) || !positive(options.ClipNorm) || !finite(options.WeightDecay) || options.WeightDecay < 0 {
		return history, fmt.Errorf("gofish: invalid GGUF training options")
	}
	time := min(m.config.ContextSize, len(ids)-1)
	budget := int64(options.BatchSize) * int64(m.config.Layers) * int64(16*time*m.config.Width+m.config.Heads*time*time)
	if budget > 16000000 {
		return history, fmt.Errorf("gofish: reduce GGUF training batch or context to fit activation budget")
	}
	p, params, err := m.parameters(true)
	if err != nil {
		return
	}
	defer func() {
		if history.Steps > 0 {
			syncErr := m.syncWeights(p)
			if err == nil {
				err = syncErr
			}
		}
	}()
	optimizer := optim.NewAdamW(params, options.LearningRate, options.WeightDecay)
	rng := rand.New(rand.NewSource(options.Seed))
	for step := 0; step < options.Steps; step++ {
		if err = ctx.Err(); err != nil {
			return
		}
		input, target := make([]int, options.BatchSize*time), make([]int, options.BatchSize*time)
		for row := 0; row < options.BatchSize; row++ {
			start := rng.Intn(len(ids) - time)
			copy(input[row*time:], ids[start:start+time])
			copy(target[row*time:], ids[start+1:start+time+1])
		}
		loss := tensor.CrossEntropy(m.forward(input, options.BatchSize, p).Reshape(-1, len(m.vocab.pieces)), target)
		value := loss.Item()
		if !finite(value) {
			return history, fmt.Errorf("gofish: non-finite GGUF training loss")
		}
		optimizer.ZeroGrad()
		loss.Backward()
		if norm := optim.ClipGradNorm(params, options.ClipNorm); !finite(norm) {
			return history, fmt.Errorf("gofish: non-finite GGUF gradients")
		}
		optimizer.Step()
		history.Steps++
		history.Losses = append(history.Losses, value)
		if options.OnStep != nil {
			options.OnStep(training.Step{Number: step + 1, Loss: value})
		}
	}
	return
}

// Save exports small models as float32 GGUF, including their tokenizer. It is
// limited to the same parameter budget as Train to bound transcoding memory.
func (m *Model) Save(path string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.CanTrain() {
		return fmt.Errorf("gofish: GGUF export is limited to 2 million parameters")
	}
	c, v := m.config, m.vocab
	metadata := map[string]any{"general.architecture": "llama", "general.name": "GoFish Llama export", "llama.context_length": uint32(c.ContextSize), "llama.embedding_length": uint32(c.Width), "llama.block_count": uint32(c.Layers), "llama.feed_forward_length": uint32(c.FeedForward), "llama.attention.head_count": uint32(c.Heads), "llama.attention.head_count_kv": uint32(c.KVHeads), "llama.attention.layer_norm_rms_epsilon": float32(c.Epsilon), "llama.rope.freq_base": float32(c.RopeBase), "llama.rope.dimension_count": uint32(c.RotaryDimensions), "tokenizer.ggml.model": v.model, "tokenizer.ggml.tokens": v.pieces, "tokenizer.ggml.bos_token_id": uint32(v.bos), "tokenizer.ggml.eos_token_id": uint32(v.eos), "tokenizer.ggml.add_bos_token": v.addBOS, "tokenizer.ggml.add_eos_token": v.addEOS, "tokenizer.ggml.add_space_prefix": v.spacePrefix}
	// Retain source licenses and provenance alongside the exported weights.
	for key, value := range m.metadata {
		if _, exists := metadata[key]; !exists {
			metadata[key] = value
		}
	}
	metadata["general.file_type"] = uint32(0)
	if len(v.scores) > 0 {
		metadata["tokenizer.ggml.scores"] = v.scores
	}
	if len(v.types) > 0 {
		metadata["tokenizer.ggml.token_type"] = v.types
	}
	if c.RopeScale != 1 {
		metadata["llama.rope.scaling.type"] = "linear"
		metadata["llama.rope.scaling.factor"] = float32(1 / c.RopeScale)
	}
	var tensors []gguf.TensorData
	for _, name := range keys(m.matrices) {
		matrix := m.matrices[name]
		values, err := matrix.Values(2000000)
		if err != nil {
			return err
		}
		tensors = append(tensors, gguf.TensorData{Name: name, Dimensions: []uint64{uint64(matrix.Columns), uint64(matrix.Rows)}, Type: gguf.F32, Data: gguf.Float32Data(values)})
	}
	for _, name := range keys(m.norms) {
		values := m.norms[name]
		tensors = append(tensors, gguf.TensorData{Name: name, Dimensions: []uint64{uint64(len(values))}, Type: gguf.F32, Data: gguf.Float32Data(values)})
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".gofish-*")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	if err = gguf.Write(f, metadata, tensors); err != nil {
		f.Close()
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	return os.Rename(f.Name(), path)
}

func (m *Model) Loss(ctx context.Context, corpus string) (float64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	ids, err := m.vocab.encode(corpus)
	if err != nil {
		return 0, err
	}
	if len(ids) < 2 {
		return 0, fmt.Errorf("gofish: evaluation text is too short")
	}
	session := m.newSession()
	total := 0.0
	for i := 0; i < len(ids)-1; i++ {
		logits, err := session.step(ctx, ids[i])
		if err != nil {
			return 0, err
		}
		peak := math.Inf(-1)
		for _, value := range logits {
			peak = max(peak, value)
		}
		sum := 0.0
		for _, value := range logits {
			sum += math.Exp(value - peak)
		}
		total += peak - logits[ids[i+1]] + math.Log(sum)
	}
	return total / float64(len(ids)-1), nil
}

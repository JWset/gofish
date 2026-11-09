// Package lm implements a small trainable character language model in pure Go.
// It uses learned embeddings, a fixed context window, a tanh hidden layer, and
// next-character cross entropy. It is a teaching model, not an instruction LLM.
package lm

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"sync"
	"unicode/utf8"

	"github.com/JWset/gofish/generation"
	"github.com/JWset/gofish/nn"
	"github.com/JWset/gofish/optim"
	"github.com/JWset/gofish/tensor"
	"github.com/JWset/gofish/training"
)

type Config struct {
	ContextSize   int   `json:"context_size"`
	EmbeddingSize int   `json:"embedding_size"`
	HiddenSize    int   `json:"hidden_size"`
	Seed          int64 `json:"seed"`
}

// Model serializes Train, Generate and Save. Direct parameter access is intended
// for custom single-threaded training loops; callers must synchronize such access.
type Model struct {
	mu             sync.Mutex
	config         Config
	vocabulary     []rune
	indices        map[rune]int
	embedding      *nn.Embedding
	hidden, output *nn.Linear
}

func defaults(c Config) Config {
	if c.ContextSize == 0 {
		c.ContextSize = 12
	}
	if c.EmbeddingSize == 0 {
		c.EmbeddingSize = 8
	}
	if c.HiddenSize == 0 {
		c.HiddenSize = 64
	}
	return c
}
func validate(c Config, vocabulary int) error {
	if c.ContextSize < 1 || c.ContextSize > 256 || c.EmbeddingSize < 1 || c.EmbeddingSize > 512 || c.HiddenSize < 1 || c.HiddenSize > 2048 || vocabulary < 2 || vocabulary > 65536 {
		return fmt.Errorf("gofish: invalid language model dimensions")
	}
	n := vocabulary*c.EmbeddingSize + c.ContextSize*c.EmbeddingSize*c.HiddenSize + c.HiddenSize + c.HiddenSize*vocabulary + vocabulary
	if n > 5000000 {
		return fmt.Errorf("gofish: native model exceeds 5 million parameter limit")
	}
	return nil
}

// New builds a vocabulary from corpus and initializes an untrained model.
// NUL is reserved for context padding; fine-tuning requires the same vocabulary.
func New(corpus string, config Config) (*Model, error) {
	if !utf8.ValidString(corpus) || utf8.RuneCountInString(corpus) < 2 {
		return nil, fmt.Errorf("gofish: corpus must contain at least two valid UTF-8 characters")
	}
	seen := map[rune]bool{}
	for _, r := range corpus {
		if r == 0 {
			return nil, fmt.Errorf("gofish: NUL is reserved for padding")
		}
		seen[r] = true
	}
	vocab := []rune{0}
	for r := range seen {
		vocab = append(vocab, r)
	}
	slices.Sort(vocab[1:])
	return newModel(vocab, defaults(config))
}
func newModel(vocab []rune, c Config) (*Model, error) {
	if err := validate(c, len(vocab)); err != nil {
		return nil, err
	}
	rng := rand.New(rand.NewSource(c.Seed))
	m := &Model{config: c, vocabulary: slices.Clone(vocab), indices: map[rune]int{}}
	for i, r := range vocab {
		m.indices[r] = i
	}
	m.embedding = nn.NewEmbedding(len(vocab), c.EmbeddingSize, rng)
	m.hidden = nn.NewLinear(c.ContextSize*c.EmbeddingSize, c.HiddenSize, rng)
	m.output = nn.NewLinear(c.HiddenSize, len(vocab), rng)
	return m, nil
}
func (m *Model) Config() Config     { return m.config }
func (m *Model) Vocabulary() string { return string(m.vocabulary[1:]) }
func (m *Model) Parameters() []*tensor.Tensor {
	p := m.embedding.Parameters()
	p = append(p, m.hidden.Parameters()...)
	return append(p, m.output.Parameters()...)
}
func (m *Model) encode(text string) ([]int, error) {
	if !utf8.ValidString(text) {
		return nil, fmt.Errorf("gofish: text must be valid UTF-8")
	}
	ids := make([]int, 0, utf8.RuneCountInString(text))
	for _, r := range text {
		id, ok := m.indices[r]
		if !ok || id == 0 {
			return nil, fmt.Errorf("gofish: character %q is outside this model's vocabulary", r)
		}
		ids = append(ids, id)
	}
	return ids, nil
}
func (m *Model) forward(contexts []int, training bool) *tensor.Tensor {
	e, w, b, v, c := m.embedding.Weight, m.hidden.Weight, m.hidden.Bias, m.output.Weight, m.output.Bias
	if !training {
		e, w, b, v, c = e.Detach(), w.Detach(), b.Detach(), v.Detach(), c.Detach()
	}
	x := e.Gather(contexts).Reshape(len(contexts)/m.config.ContextSize, m.config.ContextSize*m.config.EmbeddingSize)
	return x.MatMul(w).Add(b).Tanh().MatMul(v).Add(c)
}

type Step = training.Step
type TrainOptions = training.Options
type History = training.History

// Train performs mini-batch Adam updates, including on previously loaded models.
// Each call starts a fresh Adam optimizer. Steps are updates, not corpus epochs.
func (m *Model) Train(ctx context.Context, corpus string, options TrainOptions) (History, error) {
	if err := training.ValidateDevice(options.Device); err != nil {
		return training.History{}, err
	}
	if options.Device == training.GPU {
		return training.History{}, fmt.Errorf("gofish: GPU training currently requires NewTransformer; this model supports CPU training")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	history := History{}
	if options.Resume || options.WeightDecay != 0 {
		return history, fmt.Errorf("gofish: Resume and WeightDecay require a transformer model")
	}
	if err := ctx.Err(); err != nil {
		return history, err
	}
	ids, err := m.encode(corpus)
	if err != nil {
		return history, err
	}
	if len(ids) < 2 {
		return history, fmt.Errorf("gofish: training text is too short")
	}
	if options.Steps == 0 {
		options.Steps = 500
	}
	if options.BatchSize == 0 {
		options.BatchSize = 32
	}
	if options.LearningRate == 0 {
		options.LearningRate = 0.01
	}
	if options.ClipNorm == 0 {
		options.ClipNorm = 1
	}
	if options.Steps < 1 || options.BatchSize < 1 || options.BatchSize > 4096 || options.LearningRate <= 0 || math.IsNaN(options.LearningRate) || math.IsInf(options.LearningRate, 0) || options.ClipNorm <= 0 || math.IsNaN(options.ClipNorm) || math.IsInf(options.ClipNorm, 0) {
		return history, fmt.Errorf("gofish: invalid training options")
	}
	rng := rand.New(rand.NewSource(options.Seed))
	optimizer := optim.NewAdam(m.Parameters(), options.LearningRate)
	contextSize := m.config.ContextSize
	for step := 0; step < options.Steps; step++ {
		if err := ctx.Err(); err != nil {
			return history, err
		}
		inputs := make([]int, options.BatchSize*contextSize)
		targets := make([]int, options.BatchSize)
		for row := range targets {
			position := rng.Intn(len(ids))
			targets[row] = ids[position]
			start := max(0, position-contextSize)
			copy(inputs[row*contextSize+contextSize-(position-start):(row+1)*contextSize], ids[start:position])
		}
		loss := nn.CrossEntropy(m.forward(inputs, true), targets)
		value := loss.Item()
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return history, fmt.Errorf("gofish: non-finite training loss at step %d", step+1)
		}
		optimizer.ZeroGrad()
		loss.Backward()
		optim.ClipGradNorm(m.Parameters(), options.ClipNorm)
		optimizer.Step()
		history.Losses = append(history.Losses, value)
		history.Steps++
		if options.OnStep != nil {
			options.OnStep(Step{Number: step + 1, Loss: value})
		}
	}
	return history, nil
}

// Generate returns a continuation, excluding prompt. MaxTokens counts Unicode
// characters for this model; Ollama models use their own subword tokenization.
func (m *Model) Generate(ctx context.Context, prompt string, options ...generation.Option) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	c, err := generation.Resolve(options)
	if err != nil {
		return "", err
	}
	ids, err := m.encode(prompt)
	if err != nil {
		return "", err
	}
	window := make([]int, m.config.ContextSize)
	if len(ids) > len(window) {
		ids = ids[len(ids)-len(window):]
	}
	copy(window[len(window)-len(ids):], ids)
	rng := rand.New(rand.NewSource(c.Seed))
	collector := generation.NewCollector(c)
	for step := 0; step < c.MaxTokens; step++ {
		if err := ctx.Err(); err != nil {
			return collector.Text(), err
		}
		logits := m.forward(window, false).Data()
		selected, err := generation.Sample(logits, c, rng, func(i int) bool { return i > 0 })
		if err != nil {
			return collector.Text(), err
		}
		stop, err := collector.Feed(string(m.vocabulary[selected]))
		if err != nil {
			return collector.Text(), err
		}
		if stop {
			return collector.Text(), nil
		}
		copy(window, window[1:])
		window[len(window)-1] = selected
	}
	err = collector.Finish()
	return collector.Text(), err
}

// Loss evaluates mean cross entropy over all next-character targets without gradients.
func (m *Model) Loss(ctx context.Context, corpus string) (float64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	ids, err := m.encode(corpus)
	if err != nil {
		return 0, err
	}
	if len(ids) < 2 {
		return 0, fmt.Errorf("gofish: evaluation text is too short")
	}
	total := 0.0
	c := m.config.ContextSize
	for start := 0; start < len(ids); start += 64 {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		end := min(len(ids), start+64)
		inputs := make([]int, (end-start)*c)
		for position := start; position < end; position++ {
			n := min(c, position)
			copy(inputs[(position-start)*c+c-n:(position-start+1)*c], ids[position-n:position])
		}
		total += nn.CrossEntropy(m.forward(inputs, false), ids[start:end]).Item() * float64(end-start)
	}
	return total / float64(len(ids)), nil
}

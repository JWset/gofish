// Package transformer implements a small decoder-only language model in pure Go.
// It uses byte tokens, learned positions, causal multi-head attention, pre-layer
// normalization, GELU feed-forward layers and residual connections. It is intended
// for CPU experiments, not as a replacement for an optimized large-model runtime.
package transformer

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"unicode/utf8"

	"github.com/JWset/gofish/generation"
	"github.com/JWset/gofish/nn"
	"github.com/JWset/gofish/optim"
	"github.com/JWset/gofish/tensor"
	"github.com/JWset/gofish/tokenizer"
	"github.com/JWset/gofish/training"
)

type Config struct {
	ContextSize int   `json:"context_size"`
	Width       int   `json:"width"`
	Heads       int   `json:"heads"`
	Layers      int   `json:"layers"`
	FeedForward int   `json:"feed_forward"`
	Seed        int64 `json:"seed"`
}
type network struct {
	tokens, positions *nn.Embedding
	blocks            []*nn.TransformerBlock
	norm              *nn.LayerNorm
	head              *nn.Linear
}
type Model struct {
	mu        sync.Mutex
	config    Config
	net       *network
	optimizer *optim.Adam
	updates   int
}

func defaults(c Config) Config {
	if c.ContextSize == 0 {
		c.ContextSize = 32
	}
	if c.Width == 0 {
		c.Width = 32
	}
	if c.Heads == 0 {
		c.Heads = 4
	}
	if c.Layers == 0 {
		c.Layers = 2
	}
	if c.FeedForward == 0 {
		c.FeedForward = 4 * c.Width
	}
	return c
}
func validate(c Config) error {
	if c.ContextSize < 1 || c.ContextSize > 256 || c.Width < 1 || c.Width > 512 || c.Heads < 1 || c.Heads > 32 || c.Width%c.Heads != 0 || c.Layers < 1 || c.Layers > 8 || c.FeedForward < 1 || c.FeedForward > 2048 {
		return fmt.Errorf("gofish: invalid transformer configuration")
	}
	if parameterCount(c) > 2000000 {
		return fmt.Errorf("gofish: transformer exceeds 2 million parameter limit")
	}
	return nil
}
func parameterCount(c Config) int {
	// Token/position embeddings, decoder blocks, final norm and untied output head.
	return tokenizer.VocabularySize*c.Width + c.ContextSize*c.Width + c.Layers*(4*c.Width*c.Width+4*c.Width+2*c.Width*c.FeedForward+c.FeedForward+c.Width+4*c.Width) + 2*c.Width + c.Width*tokenizer.VocabularySize + tokenizer.VocabularySize
}

// New initializes an untrained decoder. The tokenizer represents all UTF-8 text;
// no corpus or vocabulary-building step is needed. Zero config fields use defaults.
func New(config Config) (*Model, error) {
	c := defaults(config)
	if err := validate(c); err != nil {
		return nil, err
	}
	rng := rand.New(rand.NewSource(c.Seed))
	n := &network{tokens: nn.NewEmbedding(tokenizer.VocabularySize, c.Width, rng), positions: nn.NewEmbedding(c.ContextSize, c.Width, rng), norm: nn.NewLayerNorm(c.Width), head: nn.NewLinear(c.Width, tokenizer.VocabularySize, rng)}
	for i := 0; i < c.Layers; i++ {
		n.blocks = append(n.blocks, nn.NewTransformerBlock(c.Width, c.Heads, c.FeedForward, rng))
	}
	// Small Gaussian weights keep residual paths well-scaled at initialization.
	for _, p := range n.parameters() {
		if len(p.Shape()) == 2 {
			data := make([]float64, p.Numel())
			for i := range data {
				data[i] = rng.NormFloat64() * 0.02
			}
			p.Assign(data)
		}
	}
	return &Model{config: c, net: n}, nil
}
func (m *Model) Config() Config      { return m.config }
func (m *Model) ParameterCount() int { return parameterCount(m.config) }

// Parameters and Forward expose the differentiable network for custom training.
// Synchronize direct access with other operations on the model.
func (m *Model) Parameters() []*tensor.Tensor { return m.net.parameters() }
func (m *Model) Updates() int                 { m.mu.Lock(); defer m.mu.Unlock(); return m.updates }
func (n *network) parameters() []*tensor.Tensor {
	p := n.tokens.Parameters()
	p = append(p, n.positions.Parameters()...)
	for _, b := range n.blocks {
		p = append(p, b.Parameters()...)
	}
	p = append(p, n.norm.Parameters()...)
	return append(p, n.head.Parameters()...)
}

// Forward takes flattened [batch,time] token IDs and returns [batch,time,257] logits.
func (m *Model) Forward(ids []int, batch int) *tensor.Tensor {
	if batch < 1 || len(ids) == 0 || len(ids)%batch != 0 || len(ids)/batch > m.config.ContextSize {
		panic("gofish: invalid transformer input dimensions")
	}
	return m.net.forward(ids, batch, m.config.Width)
}
func (n *network) forward(ids []int, batch, width int) *tensor.Tensor {
	return n.head.Forward(n.features(ids, batch, width))
}
func (n *network) features(ids []int, batch, width int) *tensor.Tensor {
	time := len(ids) / batch
	positions := make([]int, time)
	for i := range positions {
		positions[i] = i
	}
	x := n.tokens.Forward(ids).Reshape(batch, time, width).Add(n.positions.Forward(positions).Reshape(1, time, width))
	for _, block := range n.blocks {
		x = block.Forward(x)
	}
	return n.norm.Forward(x)
}
func detachedLinear(l *nn.Linear) *nn.Linear {
	return &nn.Linear{Weight: l.Weight.Detach(), Bias: l.Bias.Detach()}
}
func detachedNorm(l *nn.LayerNorm) *nn.LayerNorm {
	return &nn.LayerNorm{Weight: l.Weight.Detach(), Bias: l.Bias.Detach(), Epsilon: l.Epsilon}
}
func (n *network) detached() *network {
	clone := &network{tokens: &nn.Embedding{Weight: n.tokens.Weight.Detach()}, positions: &nn.Embedding{Weight: n.positions.Weight.Detach()}, norm: detachedNorm(n.norm), head: detachedLinear(n.head)}
	for _, b := range n.blocks {
		a := b.Attention
		clone.blocks = append(clone.blocks, &nn.TransformerBlock{Norm1: detachedNorm(b.Norm1), Norm2: detachedNorm(b.Norm2), Attention: &nn.MultiHeadAttention{Query: detachedLinear(a.Query), Key: detachedLinear(a.Key), Value: detachedLinear(a.Value), Projection: detachedLinear(a.Projection), Heads: a.Heads, Causal: a.Causal}, FeedForward: nn.NewSequential(detachedLinear(b.FeedForward.Layers[0].(*nn.Linear)), nn.GELU{}, detachedLinear(b.FeedForward.Layers[2].(*nn.Linear)))})
	}
	return clone
}

func tokens(text string) ([]int, error) {
	if !utf8.ValidString(text) {
		return nil, fmt.Errorf("gofish: text must be valid UTF-8")
	}
	return append([]int{tokenizer.BOS}, (tokenizer.Byte{}).Encode(text)...), nil
}
func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

// Train optimizes next-byte prediction at every position in sampled windows.
// Resume continues moments and absolute update numbering, including after Load.
// Use the same seed, batch size, corpus and other options for exact continuation.
func (m *Model) Train(ctx context.Context, corpus string, options training.Options) (training.History, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	history := training.History{}
	if err := ctx.Err(); err != nil {
		return history, err
	}
	ids, err := tokens(corpus)
	if err != nil {
		return history, err
	}
	if len(ids) < 3 {
		return history, fmt.Errorf("gofish: training text must contain at least two bytes")
	}
	if options.Steps == 0 {
		options.Steps = 500
	}
	if options.BatchSize == 0 {
		options.BatchSize = 4
	}
	if options.ClipNorm == 0 {
		options.ClipNorm = 1
	}
	if options.LearningRate == 0 {
		options.LearningRate = 0.003
		if options.Resume && m.optimizer != nil {
			options.LearningRate = m.optimizer.LearningRate
		}
	}
	if options.Resume && options.WeightDecay == 0 && m.optimizer != nil {
		options.WeightDecay = m.optimizer.WeightDecay
	}
	if options.Steps < 1 || options.BatchSize < 1 || options.BatchSize > 256 || options.LearningRate <= 0 || !finite(options.LearningRate) || options.ClipNorm <= 0 || !finite(options.ClipNorm) || options.WeightDecay < 0 || !finite(options.WeightDecay) {
		return history, fmt.Errorf("gofish: invalid transformer training options")
	}
	time := min(m.config.ContextSize, len(ids)-1)
	activationBudget := int64(options.BatchSize) * int64(m.config.Layers) * int64(16*time*m.config.Width+m.config.Heads*time*time)
	if activationBudget > 16000000 {
		return history, fmt.Errorf("gofish: training batch exceeds native activation budget; reduce batch size or context")
	}
	if options.Resume {
		if m.optimizer == nil {
			return history, fmt.Errorf("gofish: Resume requires an existing or loaded optimizer state")
		}
		m.optimizer.LearningRate = options.LearningRate
		m.optimizer.WeightDecay = options.WeightDecay
	} else {
		m.optimizer = optim.NewAdamW(m.Parameters(), options.LearningRate, options.WeightDecay)
		m.updates = 0
	}
	params := m.Parameters()
	for step := 0; step < options.Steps; step++ {
		if err := ctx.Err(); err != nil {
			return history, err
		}
		// Per-update seeds make checkpoint resume independent of rand.Source internals.
		rng := rand.New(rand.NewSource(options.Seed ^ int64(m.updates)*-7046029254386353131))
		inputs, targets := make([]int, options.BatchSize*time), make([]int, options.BatchSize*time)
		for row := 0; row < options.BatchSize; row++ {
			start := rng.Intn(len(ids) - time)
			copy(inputs[row*time:], ids[start:start+time])
			copy(targets[row*time:], ids[start+1:start+time+1])
		}
		loss := tensor.CrossEntropy(m.net.forward(inputs, options.BatchSize, m.config.Width).Reshape(-1, tokenizer.VocabularySize), targets)
		value := loss.Item()
		if !finite(value) {
			return history, fmt.Errorf("gofish: non-finite transformer loss at update %d", m.updates+1)
		}
		m.optimizer.ZeroGrad()
		loss.Backward()
		norm := optim.ClipGradNorm(params, options.ClipNorm)
		if !finite(norm) {
			return history, fmt.Errorf("gofish: non-finite transformer gradients at update %d", m.updates+1)
		}
		m.optimizer.Step()
		m.updates++
		history.Steps++
		history.Losses = append(history.Losses, value)
		if options.OnStep != nil {
			options.OnStep(training.Step{Number: step + 1, Loss: value})
		}
	}
	return history, nil
}

// Loss computes mean next-byte cross entropy with gradients disabled. Evaluation
// resets context at each non-overlapping window, including a BOS at corpus start.
func (m *Model) Loss(ctx context.Context, corpus string) (float64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	ids, err := tokens(corpus)
	if err != nil {
		return 0, err
	}
	if len(ids) < 3 {
		return 0, fmt.Errorf("gofish: evaluation text is too short")
	}
	net := m.net.detached()
	total, count := 0.0, 0
	for start := 0; start < len(ids)-1; start += m.config.ContextSize {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		end := min(len(ids)-1, start+m.config.ContextSize)
		logits := net.forward(ids[start:end], 1, m.config.Width).Reshape(-1, tokenizer.VocabularySize)
		total += tensor.CrossEntropy(logits, ids[start+1:end+1]).Item() * float64(end-start)
		count += end - start
	}
	return total / float64(count), nil
}

// Generate returns a continuation, counting byte tokens toward MaxTokens. UTF-8
// constrained decoding guarantees complete valid output, including streamed chunks.
// It recomputes attention over the retained context; there is no KV cache yet.
func (m *Model) Generate(ctx context.Context, prompt string, options ...generation.Option) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	c, err := generation.Resolve(options)
	if err != nil {
		return "", err
	}
	ids, err := tokens(prompt)
	if err != nil {
		return "", err
	}
	if err = ctx.Err(); err != nil {
		return "", err
	}
	net := m.net.detached()
	rng := rand.New(rand.NewSource(c.Seed))
	collector := generation.NewCollector(c)
	var pending []byte
	for step := 0; step < c.MaxTokens; step++ {
		if err := ctx.Err(); err != nil {
			return collector.Text(), err
		}
		if len(ids) > m.config.ContextSize {
			ids = ids[len(ids)-m.config.ContextSize:]
		}
		last := net.features(ids, 1, m.config.Width).Slice(1, len(ids)-1, len(ids))
		logits := net.head.Forward(last).Data()
		selected, err := generation.Sample(logits, c, rng, func(id int) bool {
			if id == tokenizer.BOS || !tokenizer.ValidNext(pending, byte(id)) {
				return false
			}
			lead := byte(id)
			if len(pending) > 0 {
				lead = pending[0]
			}
			width := 1
			if lead >= 0xf0 {
				width = 4
			} else if lead >= 0xe0 {
				width = 3
			} else if lead >= 0xc0 {
				width = 2
			}
			return width-len(pending)-1 <= c.MaxTokens-step-1
		})
		if err != nil {
			return collector.Text(), err
		}
		ids = append(ids, selected)
		pending = append(pending, byte(selected))
		if utf8.FullRune(pending) {
			stopped, err := collector.Feed(string(pending))
			pending = nil
			if err != nil {
				return collector.Text(), err
			}
			if stopped {
				return collector.Text(), nil
			}
		}
	}
	err = collector.Finish()
	return collector.Text(), err
}

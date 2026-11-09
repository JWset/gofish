package nn

import (
	"math/rand"

	"github.com/JWset/gofish/tensor"
)

type GELU struct{}

func (GELU) Forward(x *tensor.Tensor) *tensor.Tensor { return x.GELU() }
func (GELU) Parameters() []*tensor.Tensor            { return nil }

// LayerNorm learns a separate scale and bias for each final-dimension feature.
type LayerNorm struct {
	Weight, Bias *tensor.Tensor
	Epsilon      float64
}

func NewLayerNorm(width int) *LayerNorm {
	return &LayerNorm{Weight: tensor.Ones(width).WithGrad(), Bias: tensor.Zeros(width).WithGrad(), Epsilon: 1e-5}
}
func (l *LayerNorm) Forward(x *tensor.Tensor) *tensor.Tensor {
	shape := x.Shape()
	if len(shape) == 0 || shape[len(shape)-1] != l.Weight.Numel() {
		panic("gofish: LayerNorm feature width mismatch")
	}
	return x.LayerNorm(l.Epsilon).Mul(l.Weight).Add(l.Bias)
}
func (l *LayerNorm) Parameters() []*tensor.Tensor { return []*tensor.Tensor{l.Weight, l.Bias} }

// MultiHeadAttention takes [batch,time,width] inputs and uses causal self-attention.
type MultiHeadAttention struct {
	Query, Key, Value, Projection *Linear
	Heads                         int
	Causal                        bool
}

func NewMultiHeadAttention(width, heads int, rng *rand.Rand) *MultiHeadAttention {
	if heads < 1 || width < 1 || width%heads != 0 {
		panic("gofish: attention width must be divisible by head count")
	}
	return &MultiHeadAttention{Query: NewLinear(width, width, rng), Key: NewLinear(width, width, rng), Value: NewLinear(width, width, rng), Projection: NewLinear(width, width, rng), Heads: heads, Causal: true}
}
func (a *MultiHeadAttention) Forward(x *tensor.Tensor) *tensor.Tensor {
	return a.Projection.Forward(tensor.Attention(a.Query.Forward(x), a.Key.Forward(x), a.Value.Forward(x), a.Heads, a.Causal))
}
func (a *MultiHeadAttention) Parameters() []*tensor.Tensor {
	var p []*tensor.Tensor
	for _, m := range []*Linear{a.Query, a.Key, a.Value, a.Projection} {
		p = append(p, m.Parameters()...)
	}
	return p
}

// TransformerBlock is a decoder block with pre-normalization, causal attention,
// a GELU feed-forward network and residual connections. It has no dropout.
type TransformerBlock struct {
	Attention    *MultiHeadAttention
	Norm1, Norm2 *LayerNorm
	FeedForward  *Sequential
}

func NewTransformerBlock(width, heads, feedForward int, rng *rand.Rand) *TransformerBlock {
	return &TransformerBlock{Attention: NewMultiHeadAttention(width, heads, rng), Norm1: NewLayerNorm(width), Norm2: NewLayerNorm(width), FeedForward: NewSequential(NewLinear(width, feedForward, rng), GELU{}, NewLinear(feedForward, width, rng))}
}
func (b *TransformerBlock) Forward(x *tensor.Tensor) *tensor.Tensor {
	x = x.Add(b.Attention.Forward(b.Norm1.Forward(x)))
	return x.Add(b.FeedForward.Forward(b.Norm2.Forward(x)))
}
func (b *TransformerBlock) Parameters() []*tensor.Tensor {
	var p []*tensor.Tensor
	for _, m := range []Module{b.Norm1, b.Attention, b.Norm2, b.FeedForward} {
		p = append(p, m.Parameters()...)
	}
	return p
}

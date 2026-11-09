// Package nn provides composable neural network layers and losses.
package nn

import (
	"math"
	"math/rand"

	"github.com/JWset/gofish/tensor"
)

// Module maps tensors to tensors and exposes its trainable parameters.
type Module interface {
	Forward(*tensor.Tensor) *tensor.Tensor
	Parameters() []*tensor.Tensor
}

// Linear implements x @ Weight + Bias. Weight has shape [in, out].
type Linear struct{ Weight, Bias *tensor.Tensor }

func NewLinear(in, out int, rng *rand.Rand) *Linear {
	if in <= 0 || out <= 0 {
		panic("gofish: layer dimensions must be positive")
	}
	w := tensor.Randn(rng, in, out).Scale(math.Sqrt(1 / float64(in))).Detach().WithGrad()
	return &Linear{Weight: w, Bias: tensor.Zeros(out).WithGrad()}
}
func (l *Linear) Forward(x *tensor.Tensor) *tensor.Tensor { return x.MatMul(l.Weight).Add(l.Bias) }
func (l *Linear) Parameters() []*tensor.Tensor            { return []*tensor.Tensor{l.Weight, l.Bias} }

type Embedding struct{ Weight *tensor.Tensor }

func NewEmbedding(vocabulary, width int, rng *rand.Rand) *Embedding {
	return &Embedding{Weight: tensor.Randn(rng, vocabulary, width).Scale(0.1).Detach().WithGrad()}
}
func (e *Embedding) Forward(indices []int) *tensor.Tensor { return e.Weight.Gather(indices) }
func (e *Embedding) Parameters() []*tensor.Tensor         { return []*tensor.Tensor{e.Weight} }

type ReLU struct{}

func (ReLU) Forward(x *tensor.Tensor) *tensor.Tensor { return x.ReLU() }
func (ReLU) Parameters() []*tensor.Tensor            { return nil }

type Tanh struct{}

func (Tanh) Forward(x *tensor.Tensor) *tensor.Tensor { return x.Tanh() }
func (Tanh) Parameters() []*tensor.Tensor            { return nil }

type Sigmoid struct{}

func (Sigmoid) Forward(x *tensor.Tensor) *tensor.Tensor { return x.Sigmoid() }
func (Sigmoid) Parameters() []*tensor.Tensor            { return nil }

type Sequential struct{ Layers []Module }

func NewSequential(layers ...Module) *Sequential {
	return &Sequential{Layers: append([]Module(nil), layers...)}
}
func (s *Sequential) Forward(x *tensor.Tensor) *tensor.Tensor {
	for _, l := range s.Layers {
		x = l.Forward(x)
	}
	return x
}
func (s *Sequential) Parameters() []*tensor.Tensor {
	var parameters []*tensor.Tensor
	seen := map[*tensor.Tensor]bool{}
	for _, l := range s.Layers {
		for _, p := range l.Parameters() {
			if !seen[p] {
				parameters = append(parameters, p)
				seen[p] = true
			}
		}
	}
	return parameters
}

func CrossEntropy(logits *tensor.Tensor, targets []int) *tensor.Tensor {
	return tensor.CrossEntropy(logits, targets)
}
func MSE(prediction, target *tensor.Tensor) *tensor.Tensor { return tensor.MSE(prediction, target) }

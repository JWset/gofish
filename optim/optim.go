// Package optim provides SGD, Adam, and global gradient norm clipping.
package optim

import (
	"math"

	"github.com/JWset/gofish/tensor"
)

func parameters(params []*tensor.Tensor) []*tensor.Tensor {
	out := make([]*tensor.Tensor, 0, len(params))
	seen := map[*tensor.Tensor]bool{}
	for _, p := range params {
		if p == nil || !p.RequiresGrad() {
			panic("gofish: optimizer requires trainable parameters")
		}
		if !seen[p] {
			out = append(out, p)
			seen[p] = true
		}
	}
	return out
}

type SGD struct {
	params       []*tensor.Tensor
	LearningRate float64
}

func NewSGD(params []*tensor.Tensor, learningRate float64) *SGD {
	validRate(learningRate)
	return &SGD{parameters(params), learningRate}
}
func (s *SGD) ZeroGrad() {
	for _, p := range s.params {
		p.ZeroGrad()
	}
}
func (s *SGD) Step() {
	validRate(s.LearningRate)
	for _, p := range s.params {
		p.Update(func(_ int, v, g float64) float64 { return v - s.LearningRate*g })
	}
}

type adamState struct {
	first, second []float64
	step          int
}
type Adam struct {
	params       []*tensor.Tensor
	states       []adamState
	LearningRate float64
}

func validRate(rate float64) {
	if rate <= 0 || math.IsNaN(rate) || math.IsInf(rate, 0) {
		panic("gofish: learning rate must be finite and positive")
	}
}

// NewAdam uses beta1=0.9, beta2=0.999, epsilon=1e-8 and bias correction.
// Optimizer state is in memory; model checkpoints contain weights, not Adam moments.
func NewAdam(params []*tensor.Tensor, learningRate float64) *Adam {
	validRate(learningRate)
	params = parameters(params)
	a := &Adam{params: params, LearningRate: learningRate, states: make([]adamState, len(params))}
	for i, p := range params {
		a.states[i] = adamState{first: make([]float64, p.Numel()), second: make([]float64, p.Numel())}
	}
	return a
}
func (a *Adam) ZeroGrad() {
	for _, p := range a.params {
		p.ZeroGrad()
	}
}
func (a *Adam) Step() {
	validRate(a.LearningRate)
	for j, p := range a.params {
		if p.Grad() == nil {
			continue
		}
		s := &a.states[j]
		s.step++
		c1, c2 := 1-math.Pow(0.9, float64(s.step)), 1-math.Pow(0.999, float64(s.step))
		p.Update(func(i int, v, g float64) float64 {
			s.first[i] = 0.9*s.first[i] + 0.1*g
			s.second[i] = 0.999*s.second[i] + 0.001*g*g
			return v - a.LearningRate*(s.first[i]/c1)/(math.Sqrt(s.second[i]/c2)+1e-8)
		})
	}
}

// ClipGradNorm returns the original global L2 norm and rescales gradients if needed.
func ClipGradNorm(params []*tensor.Tensor, maxNorm float64) float64 {
	validRate(maxNorm)
	params = parameters(params)
	norm := 0.0
	for _, p := range params {
		for _, g := range p.Grad() {
			norm = math.Hypot(norm, g)
		}
	}
	if norm > maxNorm {
		for _, p := range params {
			p.ScaleGrad(maxNorm / norm)
		}
	}
	return norm
}

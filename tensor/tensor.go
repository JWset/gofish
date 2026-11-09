// Package tensor provides dense float64 CPU tensors and reverse-mode automatic
// differentiation. Shape and arithmetic errors panic, like out-of-range Go slices.
// Tensors are not safe for concurrent mutation or backward passes.
package tensor

import (
	"fmt"
	"math/rand"
	"slices"
)

// Tensor owns contiguous row-major storage. Data, Shape and Grad return copies.
// A zero-dimensional tensor is a scalar; zero-sized dimensions are unsupported.
type Tensor struct {
	data         []float64
	shape        []int
	grad         []float64
	requiresGrad bool
	parents      []*Tensor
	versions     []uint64
	backward     func()
	version      uint64
}

func size(shape []int) int {
	n := 1
	for _, d := range shape {
		if d <= 0 || n > int(^uint(0)>>1)/d {
			panic("gofish: invalid tensor shape")
		}
		n *= d
	}
	return n
}

// New copies data into a tensor of the supplied shape.
func New(data []float64, shape ...int) *Tensor {
	if size(shape) != len(data) {
		panic("gofish: data length does not match shape")
	}
	return &Tensor{data: slices.Clone(data), shape: slices.Clone(shape)}
}

func Scalar(value float64) *Tensor { return New([]float64{value}) }
func Zeros(shape ...int) *Tensor   { return New(make([]float64, size(shape)), shape...) }
func Ones(shape ...int) *Tensor {
	t := Zeros(shape...)
	for i := range t.data {
		t.data[i] = 1
	}
	return t
}

// Randn uses an explicit random source, so callers can reproduce initialization.
func Randn(rng *rand.Rand, shape ...int) *Tensor {
	t := Zeros(shape...)
	for i := range t.data {
		t.data[i] = rng.NormFloat64()
	}
	return t
}

// WithGrad makes a leaf tensor trainable. It cannot alter an existing graph.
func (t *Tensor) WithGrad() *Tensor {
	if len(t.parents) > 0 {
		panic("gofish: WithGrad requires a leaf tensor")
	}
	t.requiresGrad = true
	return t
}
func (t *Tensor) RequiresGrad() bool { return t.requiresGrad }
func (t *Tensor) Shape() []int       { return slices.Clone(t.shape) }
func (t *Tensor) Data() []float64    { return slices.Clone(t.data) }
func (t *Tensor) Grad() []float64    { return slices.Clone(t.grad) }
func (t *Tensor) Numel() int         { return len(t.data) }
func (t *Tensor) Item() float64 {
	if len(t.data) != 1 {
		panic("gofish: Item requires one element")
	}
	return t.data[0]
}
func (t *Tensor) String() string { return fmt.Sprintf("Tensor(shape=%v, data=%v)", t.shape, t.data) }
func (t *Tensor) ZeroGrad()      { clear(t.grad) }

// ScaleGrad rescales an existing gradient, for example for gradient clipping.
func (t *Tensor) ScaleGrad(scale float64) {
	for i := range t.grad {
		t.grad[i] *= scale
	}
}
func (t *Tensor) Detach() *Tensor { return New(t.data, t.shape...) }

// Assign updates a leaf's values without changing its gradient. Mutation of a
// value needed by an outstanding backward pass is detected by Backward.
func (t *Tensor) Assign(data []float64) {
	if len(t.parents) > 0 || len(data) != len(t.data) {
		panic("gofish: Assign requires a leaf and matching data")
	}
	copy(t.data, data)
	t.version++
}

// Update applies an optimizer update to a leaf with a computed gradient.
func (t *Tensor) Update(f func(index int, value, gradient float64) float64) {
	if len(t.parents) > 0 {
		panic("gofish: Update requires a leaf")
	}
	if t.grad == nil {
		return
	}
	for i, v := range t.data {
		t.data[i] = f(i, v, t.grad[i])
	}
	t.version++
}

func node(data []float64, shape []int, parents ...*Tensor) *Tensor {
	t := &Tensor{data: data, shape: slices.Clone(shape)}
	for _, p := range parents {
		t.requiresGrad = t.requiresGrad || p.requiresGrad
	}
	if t.requiresGrad {
		t.parents = parents
		t.versions = make([]uint64, len(parents))
		for i, p := range parents {
			t.versions[i] = p.version
		}
	}
	return t
}

func (t *Tensor) accumulate(i int, value float64) {
	if !t.requiresGrad {
		return
	}
	if t.grad == nil {
		t.grad = make([]float64, len(t.data))
	}
	t.grad[i] += value
}

// Backward computes a vector-Jacobian product. Scalars use an implicit seed of
// one; non-scalars require a seed matching their shape. Leaf gradients accumulate
// across calls, while intermediate gradients are reset on every traversal.
// Graphs remain reusable until a parent is mutated. Higher derivatives are unsupported.
func (t *Tensor) Backward(seed ...*Tensor) {
	if !t.requiresGrad {
		panic("gofish: tensor does not require gradients")
	}
	if len(seed) > 1 {
		panic("gofish: Backward accepts at most one seed")
	}
	var initial []float64
	if len(seed) == 1 {
		if !slices.Equal(seed[0].shape, t.shape) {
			panic("gofish: gradient seed shape mismatch")
		}
		initial = seed[0].data
	} else {
		if len(t.data) != 1 {
			panic("gofish: non-scalar Backward requires a seed")
		}
		initial = []float64{1}
	}
	// Explicit post-order traversal also handles shared branches and deep graphs.
	type frame struct {
		t       *Tensor
		visited bool
	}
	stack := []frame{{t, false}}
	seen := map[*Tensor]bool{}
	topo := []*Tensor{}
	for len(stack) > 0 {
		f := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if f.visited {
			topo = append(topo, f.t)
			continue
		}
		if seen[f.t] {
			continue
		}
		seen[f.t] = true
		stack = append(stack, frame{f.t, true})
		for i, p := range f.t.parents {
			if p.version != f.t.versions[i] {
				panic("gofish: tensor modified after forward pass")
			}
			if p.requiresGrad {
				stack = append(stack, frame{p, false})
			}
		}
	}
	for _, v := range topo {
		if len(v.parents) > 0 {
			v.grad = make([]float64, len(v.data))
		}
	}
	for i, v := range initial {
		t.accumulate(i, v)
	}
	for i := len(topo) - 1; i >= 0; i-- {
		if topo[i].backward != nil {
			topo[i].backward()
		}
	}
}

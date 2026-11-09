package tensor_test

import (
	"math"
	"slices"
	"testing"

	"github.com/JWset/gofish/tensor"
)

func near(t *testing.T, got, want, tolerance float64) {
	t.Helper()
	if math.IsNaN(got) || math.Abs(got-want) > tolerance {
		t.Fatalf("got %.10g, want %.10g (tolerance %g)", got, want, tolerance)
	}
}
func panics(t *testing.T, f func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	f()
}

// Numerical differences exercise the complete graph, including shared branches
// and broadcasting reductions, independently of the handwritten derivatives.
func TestNumericalGradients(t *testing.T) {
	tests := []struct {
		name   string
		a, b   []float64
		as, bs []int
		f      func(*tensor.Tensor, *tensor.Tensor) *tensor.Tensor
	}{
		{"broadcast add", []float64{0.3, -0.4, 0.7, 0.2, 0.8, -0.9}, []float64{0.2, 0.4, 0.6}, []int{2, 3}, []int{3}, func(a, b *tensor.Tensor) *tensor.Tensor { return a.Add(b).Pow(2).Mean() }},
		{"rank broadcast", []float64{0.3, -0.4}, []float64{0.2, 0.4, 0.6}, []int{2, 1, 1}, []int{1, 3, 1}, func(a, b *tensor.Tensor) *tensor.Tensor { return a.Mul(b).Sum() }},
		{"subtract divide", []float64{0.3, -0.4, 0.7, 0.2}, []float64{0.8, 1.4}, []int{2, 2}, []int{2, 1}, func(a, b *tensor.Tensor) *tensor.Tensor { return a.Sub(b).Div(b).Pow(2).Mean() }},
		{"matmul", []float64{0.3, -0.4, 0.7, 0.2, 0.8, -0.9}, []float64{0.2, 0.4, 0.6, -0.5, 0.8, 0.9}, []int{2, 3}, []int{3, 2}, func(a, b *tensor.Tensor) *tensor.Tensor { return a.MatMul(b).Tanh().Pow(2).Mean() }},
		{"shared diamond", []float64{0.3, -0.4}, []float64{0.2, 0.4}, []int{2}, []int{2}, func(a, b *tensor.Tensor) *tensor.Tensor { x := a.Mul(b); return x.Pow(2).Add(x.Mul(a)).Sum() }},
		{"activations", []float64{0.3, -0.4}, []float64{0.2, 0.4}, []int{2}, []int{2}, func(a, b *tensor.Tensor) *tensor.Tensor {
			return a.Exp().Log().Tanh().Add(a.Sigmoid()).Add(a.ReLU()).Add(b.Neg()).Sum()
		}},
		{"softmax", []float64{0.3, -0.4, 0.7, 0.2, 0.8, -0.9}, []float64{0.2, 0.4, 0.6}, []int{2, 3}, []int{3}, func(a, b *tensor.Tensor) *tensor.Tensor { return a.Softmax().Mul(b).Sum() }},
		{"reshape transpose", []float64{0.3, -0.4, 0.7, 0.2, 0.8, -0.9}, []float64{0.2, 0.4}, []int{2, 3}, []int{2}, func(a, b *tensor.Tensor) *tensor.Tensor {
			return a.Reshape(3, -1).Transpose().Reshape(3, 2).Mul(b).Pow(2).Sum()
		}},
		{"cross entropy", []float64{0.3, -0.4, 0.7, 0.2, 0.8, -0.9}, []float64{0.2, 0.4, 0.6}, []int{2, 3}, []int{3}, func(a, b *tensor.Tensor) *tensor.Tensor { return tensor.CrossEntropy(a.Add(b), []int{2, 0}) }},
		{"embedding repeats", []float64{0.3, -0.4, 0.7, 0.2, 0.8, -0.9}, []float64{0.2, 0.4}, []int{3, 2}, []int{2}, func(a, b *tensor.Tensor) *tensor.Tensor { return a.Gather([]int{2, 0, 2}).Mul(b).Pow(2).Sum() }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a, b := tensor.New(tt.a, tt.as...).WithGrad(), tensor.New(tt.b, tt.bs...).WithGrad()
			tt.f(a, b).Backward()
			for which, values := range [][]float64{tt.a, tt.b} {
				grad := a.Grad()
				if which == 1 {
					grad = b.Grad()
				}
				for i := range values {
					plus, minus := slices.Clone(values), slices.Clone(values)
					plus[i] += 1e-6
					minus[i] -= 1e-6
					x, y := tensor.New(tt.a, tt.as...), tensor.New(tt.b, tt.bs...)
					var hi, lo float64
					if which == 0 {
						hi = tt.f(tensor.New(plus, tt.as...), y).Item()
						lo = tt.f(tensor.New(minus, tt.as...), y).Item()
					} else {
						hi = tt.f(x, tensor.New(plus, tt.bs...)).Item()
						lo = tt.f(x, tensor.New(minus, tt.bs...)).Item()
					}
					near(t, grad[i], (hi-lo)/2e-6, 2e-5)
				}
			}
		})
	}
}

func TestBackwardAccumulationAndSeed(t *testing.T) {
	x := tensor.New([]float64{2, 3}, 2).WithGrad()
	y := x.Mul(x).Sum()
	y.Backward()
	y.Backward()
	if !slices.Equal(x.Grad(), []float64{8, 12}) {
		t.Fatal(x.Grad())
	}
	x.ZeroGrad()
	y.Backward()
	if !slices.Equal(x.Grad(), []float64{4, 6}) {
		t.Fatal(x.Grad())
	}
	x.ZeroGrad()
	x.Mul(x).Backward(tensor.New([]float64{2, 4}, 2))
	if !slices.Equal(x.Grad(), []float64{8, 24}) {
		t.Fatal(x.Grad())
	}
	panics(t, func() { x.Mul(x).Backward() })
	panics(t, func() { x.Mul(x).Backward(tensor.Ones(1, 2)) })
}
func TestMutationDetectionAndOwnership(t *testing.T) {
	data := []float64{2, 3}
	x := tensor.New(data, 2).WithGrad()
	data[0] = 99
	x.Data()[0] = 99
	x.Shape()[0] = 99
	if x.Data()[0] != 2 || x.Shape()[0] != 2 {
		t.Fatal("tensor aliases external storage")
	}
	c := tensor.Ones(2)
	y := x.Mul(c).Sum()
	c.Assign([]float64{2, 2})
	panics(t, func() { y.Backward() })
	if x.Detach().RequiresGrad() {
		t.Fatal("detach tracks gradients")
	}
}
func TestStableCrossEntropy(t *testing.T) {
	x := tensor.New([]float64{10000, 9999, -10000}, 1, 3).WithGrad()
	loss := tensor.CrossEntropy(x, []int{0})
	near(t, loss.Item(), math.Log1p(math.Exp(-1)), 1e-10)
	loss.Backward()
	for _, g := range x.Grad() {
		if math.IsNaN(g) || math.IsInf(g, 0) {
			t.Fatal("nonfinite gradient")
		}
	}
	near(t, tensor.Scalar(0).WithGrad().Pow(0).Item(), 1, 0)
}
func TestDeepGraph(t *testing.T) {
	x := tensor.Scalar(1).WithGrad()
	y := x
	for i := 0; i < 20000; i++ {
		y = y.Add(tensor.Scalar(1))
	}
	y.Backward()
	near(t, x.Grad()[0], 1, 0)
}
func TestInvalidShapes(t *testing.T) {
	for _, f := range []func(){func() { tensor.New([]float64{1}, 2) }, func() { tensor.Zeros(0) }, func() { tensor.Zeros(2).Add(tensor.Zeros(3)) }, func() { tensor.Zeros(2, 3).MatMul(tensor.Zeros(2, 3)) }, func() { tensor.Zeros(2, 3).Reshape(-1, -1) }, func() { tensor.Zeros(2, 3).Gather([]int{2}) }, func() { tensor.CrossEntropy(tensor.Zeros(2, 3), []int{0, 3}) }} {
		panics(t, f)
	}
}

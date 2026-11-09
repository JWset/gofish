package tensor_test

import (
	"math/rand"
	"slices"
	"testing"

	"github.com/JWset/gofish/tensor"
)

func checkGradients(t *testing.T, values [][]float64, shapes [][]int, eval func([]*tensor.Tensor) *tensor.Tensor) {
	t.Helper()
	params := make([]*tensor.Tensor, len(values))
	for i, data := range values {
		params[i] = tensor.New(data, shapes[i]...).WithGrad()
	}
	eval(params).Backward()
	for parameter, data := range values {
		grad := params[parameter].Grad()
		for j := range data {
			plus, minus := make([]*tensor.Tensor, len(values)), make([]*tensor.Tensor, len(values))
			for i, source := range values {
				hi, lo := slices.Clone(source), slices.Clone(source)
				if i == parameter {
					hi[j] += 1e-6
					lo[j] -= 1e-6
				}
				plus[i] = tensor.New(hi, shapes[i]...)
				minus[i] = tensor.New(lo, shapes[i]...)
			}
			want := (eval(plus).Item() - eval(minus).Item()) / 2e-6
			near(t, grad[j], want, 3e-5)
		}
	}
}
func TestTransformerPrimitiveGradients(t *testing.T) {
	data := []float64{0.2, -0.8, 1.1, 0.6, 0.3, -0.2, 1.3, -0.5}
	t.Run("GELU", func(t *testing.T) {
		checkGradients(t, [][]float64{data}, [][]int{{2, 4}}, func(p []*tensor.Tensor) *tensor.Tensor { return p[0].GELU().Pow(2).Mean() })
	})
	t.Run("LayerNorm", func(t *testing.T) {
		checkGradients(t, [][]float64{data}, [][]int{{2, 4}}, func(p []*tensor.Tensor) *tensor.Tensor {
			return p[0].LayerNorm(1e-5).Mul(tensor.New([]float64{1, 2, 3, 4}, 4)).Tanh().Sum()
		})
	})
	t.Run("PermuteSlice", func(t *testing.T) {
		checkGradients(t, [][]float64{data}, [][]int{{2, 2, 2}}, func(p []*tensor.Tensor) *tensor.Tensor { return p[0].Permute(2, 0, 1).Slice(-1, 0, 1).Pow(2).Sum() })
	})
	t.Run("BatchedMatMul", func(t *testing.T) {
		checkGradients(t, [][]float64{data, []float64{0.2, 0.5, -0.7, 0.4}}, [][]int{{2, 2, 2}, {2, 2}}, func(p []*tensor.Tensor) *tensor.Tensor { return p[0].MatMul(p[1]).Tanh().Sum() })
	})
	t.Run("BroadcastBatchedMatMul", func(t *testing.T) {
		checkGradients(t, [][]float64{data, []float64{0.2, 0.5, -0.7, 0.4}}, [][]int{{2, 1, 2, 2}, {1, 2, 2, 1}}, func(p []*tensor.Tensor) *tensor.Tensor { return p[0].MatMul(p[1]).Pow(2).Mean() })
	})
	rng := rand.New(rand.NewSource(42))
	values := make([][]float64, 3)
	for i := range values {
		values[i] = tensor.Randn(rng, 2, 3, 4).Data()
	}
	for _, causal := range []bool{true, false} {
		name := "Attention"
		if causal {
			name = "CausalAttention"
		}
		t.Run(name, func(t *testing.T) {
			checkGradients(t, values, [][]int{{2, 3, 4}, {2, 3, 4}, {2, 3, 4}}, func(p []*tensor.Tensor) *tensor.Tensor {
				return tensor.Attention(p[0], p[1], p[2], 2, causal).Pow(2).Mean()
			})
		})
	}
}

func TestAttentionCannotSeeFuture(t *testing.T) {
	q := tensor.Ones(1, 3, 4).WithGrad()
	k := tensor.Ones(1, 3, 4).WithGrad()
	v := tensor.New([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 1, 3, 4).WithGrad()
	before := tensor.Attention(q, k, v, 2, true).Data()
	changed := v.Data()
	for i := 8; i < 12; i++ {
		changed[i] += 1000
	}
	after := tensor.Attention(q, k, tensor.New(changed, 1, 3, 4), 2, true).Data()
	if !slices.Equal(before[:8], after[:8]) {
		t.Fatal("future token leaked into past outputs")
	}
	seed := tensor.New([]float64{1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}, 1, 3, 4)
	tensor.Attention(q, k, v, 2, true).Backward(seed)
	for _, p := range []*tensor.Tensor{q, k, v} {
		for _, g := range p.Grad()[4:] {
			if g != 0 {
				t.Fatal("future token has gradient from first output")
			}
		}
	}
}
func TestTransformerPrimitiveShapeErrors(t *testing.T) {
	for _, f := range []func(){func() { tensor.Ones(2, 2).Permute(0, 0) }, func() { tensor.Ones(2, 2).Slice(0, 1, 1) }, func() { tensor.Scalar(1).LayerNorm(1e-5) }, func() { tensor.Attention(tensor.Ones(1, 2, 3), tensor.Ones(1, 2, 3), tensor.Ones(1, 2, 3), 2, true) }} {
		panics(t, f)
	}
}

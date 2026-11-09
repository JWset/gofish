package tensor_test

import (
	"github.com/JWset/gofish/tensor"
	"testing"
)

func TestLlamaPrimitiveGradients(t *testing.T) {
	data := []float64{0.2, -0.8, 1.1, 0.6, 0.3, -0.2, 1.3, -0.5}
	for _, test := range []struct {
		name string
		f    func(*tensor.Tensor) *tensor.Tensor
	}{
		{"RMSNorm", func(x *tensor.Tensor) *tensor.Tensor { return x.RMSNorm(1e-5).Tanh().Sum() }},
		{"SiLU", func(x *tensor.Tensor) *tensor.Tensor { return x.SiLU().Pow(2).Sum() }},
		{"RoPE", func(x *tensor.Tensor) *tensor.Tensor { return x.RoPE(2, 2, 3, 10000, 0.5).Tanh().Sum() }},
		{"GroupedHeads", func(x *tensor.Tensor) *tensor.Tensor { return x.RepeatHeads(2, 2).Pow(2).Sum() }},
	} {
		t.Run(test.name, func(t *testing.T) {
			checkGradients(t, [][]float64{data}, [][]int{{1, 2, 4}}, func(p []*tensor.Tensor) *tensor.Tensor { return test.f(p[0]) })
		})
	}
}

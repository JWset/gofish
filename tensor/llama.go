package tensor

import "math"

func (a *Tensor) SiLU() *Tensor {
	sigmoid := func(x float64) float64 {
		if x >= 0 {
			return 1 / (1 + math.Exp(-x))
		}
		e := math.Exp(x)
		return e / (1 + e)
	}
	return unary(a, func(x float64) float64 { return x * sigmoid(x) }, func(x float64) float64 { s := sigmoid(x); return s * (1 + x*(1-s)) })
}

// RMSNorm normalizes the last dimension by its root mean square.
func (a *Tensor) RMSNorm(epsilon float64) *Tensor {
	if len(a.shape) == 0 || epsilon <= 0 || math.IsNaN(epsilon) || math.IsInf(epsilon, 0) {
		panic("gofish: invalid RMSNorm input")
	}
	n := a.shape[len(a.shape)-1]
	data := make([]float64, len(a.data))
	inverse := make([]float64, len(data)/n)
	for row := range inverse {
		start := row * n
		sum := 0.0
		for _, x := range a.data[start : start+n] {
			sum += x * x
		}
		inverse[row] = 1 / math.Sqrt(sum/float64(n)+epsilon)
		for j := 0; j < n; j++ {
			data[start+j] = a.data[start+j] * inverse[row]
		}
	}
	out := node(data, a.shape, a)
	if out.requiresGrad {
		out.backward = func() {
			for row, inv := range inverse {
				start := row * n
				dot := 0.0
				for j := 0; j < n; j++ {
					dot += out.grad[start+j] * data[start+j]
				}
				for j := 0; j < n; j++ {
					a.accumulate(start+j, inv*(out.grad[start+j]-data[start+j]*dot/float64(n)))
				}
			}
		}
	}
	return out
}

// RoPE rotates adjacent feature pairs within each head, using GGML's normal
// (interleaved-pair) convention on [batch,time,width]. Offset is the first position.
func (a *Tensor) RoPE(heads, dimensions, offset int, base, scale float64) *Tensor {
	if len(a.shape) != 3 || heads < 1 || a.shape[2]%heads != 0 || dimensions < 2 || dimensions%2 != 0 || dimensions > a.shape[2]/heads || offset < 0 || base <= 0 || scale <= 0 || math.IsNaN(base) || math.IsInf(base, 0) || math.IsNaN(scale) || math.IsInf(scale, 0) {
		panic("gofish: invalid rotary embedding configuration")
	}
	b, t, c := a.shape[0], a.shape[1], a.shape[2]
	width := c / heads
	data := a.Data()
	rotate := func(src, dst []float64, direction float64) {
		for batch := 0; batch < b; batch++ {
			for pos := 0; pos < t; pos++ {
				for head := 0; head < heads; head++ {
					start := (batch*t+pos)*c + head*width
					for j := 0; j < dimensions; j += 2 {
						angle := float64(offset+pos) * scale * math.Pow(base, -float64(j)/float64(dimensions))
						s, co := math.Sincos(angle)
						s *= direction
						x, y := src[start+j], src[start+j+1]
						dst[start+j] = x*co - y*s
						dst[start+j+1] = x*s + y*co
					}
				}
			}
		}
	}
	rotate(a.data, data, 1)
	out := node(data, a.shape, a)
	if out.requiresGrad {
		out.backward = func() {
			grad := append([]float64(nil), out.grad...)
			rotate(out.grad, grad, -1)
			for i, g := range grad {
				a.accumulate(i, g)
			}
		}
	}
	return out
}

// RepeatHeads repeats each contiguous attention head, for grouped-query attention.
func (a *Tensor) RepeatHeads(headWidth, repeats int) *Tensor {
	if len(a.shape) != 3 || headWidth < 1 || a.shape[2]%headWidth != 0 || repeats < 1 {
		panic("gofish: invalid grouped-query head shape")
	}
	shape := a.Shape()
	shape[2] *= repeats
	data := make([]float64, size(shape))
	mapping := make([]int, len(data))
	original := a.shape[2]
	for i := range data {
		col := i % shape[2]
		mapping[i] = (i/shape[2])*original + (col/headWidth/repeats)*headWidth + col%headWidth
		data[i] = a.data[mapping[i]]
	}
	out := node(data, shape, a)
	if out.requiresGrad {
		out.backward = func() {
			for i, g := range out.grad {
				a.accumulate(mapping[i], g)
			}
		}
	}
	return out
}

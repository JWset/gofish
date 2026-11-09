package tensor

import (
	"math"
	"slices"
)

func strides(shape []int) []int {
	s := make([]int, len(shape))
	n := 1
	for i := len(shape) - 1; i >= 0; i-- {
		s[i] = n
		n *= shape[i]
	}
	return s
}

func broadcast(a, b []int) []int {
	n := max(len(a), len(b))
	shape := make([]int, n)
	for i := 0; i < n; i++ {
		x, y := 1, 1
		if j := i - (n - len(a)); j >= 0 {
			x = a[j]
		}
		if j := i - (n - len(b)); j >= 0 {
			y = b[j]
		}
		if x != y && x != 1 && y != 1 {
			panic("gofish: incompatible broadcast shapes")
		}
		shape[i] = max(x, y)
	}
	return shape
}

func offsets(shape, parent []int) []int {
	s := strides(shape)
	p := strides(parent)
	out := make([]int, size(shape))
	for i := range out {
		for j, d := range parent {
			if d != 1 {
				axis := len(shape) - len(parent) + j
				out[i] += (i / s[axis] % shape[axis]) * p[j]
			}
		}
	}
	return out
}

func binary(a, b *Tensor, f func(float64, float64) float64, derivative func(float64, float64) (float64, float64)) *Tensor {
	shape := broadcast(a.shape, b.shape)
	ai, bi := offsets(shape, a.shape), offsets(shape, b.shape)
	data := make([]float64, size(shape))
	for i := range data {
		data[i] = f(a.data[ai[i]], b.data[bi[i]])
	}
	out := node(data, shape, a, b)
	if out.requiresGrad {
		out.backward = func() {
			for i, g := range out.grad {
				da, db := derivative(a.data[ai[i]], b.data[bi[i]])
				a.accumulate(ai[i], g*da)
				b.accumulate(bi[i], g*db)
			}
		}
	}
	return out
}

func (a *Tensor) Add(b *Tensor) *Tensor {
	return binary(a, b, func(x, y float64) float64 { return x + y }, func(x, y float64) (float64, float64) { return 1, 1 })
}
func (a *Tensor) Sub(b *Tensor) *Tensor {
	return binary(a, b, func(x, y float64) float64 { return x - y }, func(x, y float64) (float64, float64) { return 1, -1 })
}
func (a *Tensor) Mul(b *Tensor) *Tensor {
	return binary(a, b, func(x, y float64) float64 { return x * y }, func(x, y float64) (float64, float64) { return y, x })
}
func (a *Tensor) Div(b *Tensor) *Tensor {
	return binary(a, b, func(x, y float64) float64 { return x / y }, func(x, y float64) (float64, float64) { return 1 / y, -x / (y * y) })
}
func (a *Tensor) Scale(s float64) *Tensor { return a.Mul(Scalar(s)) }
func (a *Tensor) Neg() *Tensor            { return a.Scale(-1) }

func unary(a *Tensor, f, derivative func(float64) float64) *Tensor {
	data := make([]float64, len(a.data))
	for i, x := range a.data {
		data[i] = f(x)
	}
	out := node(data, a.shape, a)
	if out.requiresGrad {
		out.backward = func() {
			for i, g := range out.grad {
				a.accumulate(i, g*derivative(a.data[i]))
			}
		}
	}
	return out
}
func (a *Tensor) Exp() *Tensor { return unary(a, math.Exp, math.Exp) }
func (a *Tensor) Log() *Tensor { return unary(a, math.Log, func(x float64) float64 { return 1 / x }) }
func (a *Tensor) Pow(p float64) *Tensor {
	return unary(a, func(x float64) float64 { return math.Pow(x, p) }, func(x float64) float64 {
		if p == 0 {
			return 0
		}
		return p * math.Pow(x, p-1)
	})
}
func (a *Tensor) Tanh() *Tensor {
	return unary(a, math.Tanh, func(x float64) float64 { v := math.Tanh(x); return 1 - v*v })
}
func (a *Tensor) ReLU() *Tensor {
	return unary(a, func(x float64) float64 { return max(x, 0) }, func(x float64) float64 {
		if x > 0 {
			return 1
		}
		return 0
	})
}
func (a *Tensor) Sigmoid() *Tensor {
	sig := func(x float64) float64 {
		if x >= 0 {
			return 1 / (1 + math.Exp(-x))
		}
		e := math.Exp(x)
		return e / (1 + e)
	}
	return unary(a, sig, func(x float64) float64 { s := sig(x); return s * (1 - s) })
}

func (a *Tensor) Sum() *Tensor {
	v := 0.0
	for _, x := range a.data {
		v += x
	}
	out := node([]float64{v}, nil, a)
	if out.requiresGrad {
		out.backward = func() {
			for i := range a.data {
				a.accumulate(i, out.grad[0])
			}
		}
	}
	return out
}
func (a *Tensor) Mean() *Tensor { return a.Sum().Scale(1 / float64(len(a.data))) }

// Reshape copies storage and preserves differentiation. One dimension may be -1.
func (a *Tensor) Reshape(shape ...int) *Tensor {
	shape = slices.Clone(shape)
	inferred := -1
	known := 1
	for i, d := range shape {
		if d == -1 {
			if inferred != -1 {
				panic("gofish: multiple inferred dimensions")
			}
			inferred = i
		} else {
			known *= size([]int{d})
		}
	}
	if inferred >= 0 {
		if len(a.data)%known != 0 {
			panic("gofish: invalid reshape")
		}
		shape[inferred] = len(a.data) / known
	}
	if size(shape) != len(a.data) {
		panic("gofish: invalid reshape")
	}
	out := node(slices.Clone(a.data), shape, a)
	if out.requiresGrad {
		out.backward = func() {
			for i, g := range out.grad {
				a.accumulate(i, g)
			}
		}
	}
	return out
}

// Transpose swaps the two dimensions of a matrix.
func (a *Tensor) Transpose() *Tensor {
	if len(a.shape) != 2 {
		panic("gofish: Transpose requires a matrix")
	}
	m, n := a.shape[0], a.shape[1]
	data := make([]float64, len(a.data))
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			data[j*m+i] = a.data[i*n+j]
		}
	}
	out := node(data, []int{n, m}, a)
	if out.requiresGrad {
		out.backward = func() {
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					a.accumulate(i*n+j, out.grad[j*m+i])
				}
			}
		}
	}
	return out
}

// MatMul multiplies [rows, inner] by [inner, columns]. Batched MatMul is not yet supported.
func (a *Tensor) MatMul(b *Tensor) *Tensor {
	if len(a.shape) != 2 || len(b.shape) != 2 || a.shape[1] != b.shape[0] {
		panic("gofish: MatMul requires compatible matrices")
	}
	m, k, n := a.shape[0], a.shape[1], b.shape[1]
	data := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for p := 0; p < k; p++ {
			v := a.data[i*k+p]
			for j := 0; j < n; j++ {
				data[i*n+j] += v * b.data[p*n+j]
			}
		}
	}
	out := node(data, []int{m, n}, a, b)
	if out.requiresGrad {
		out.backward = func() {
			for i := 0; i < m; i++ {
				for p := 0; p < k; p++ {
					for j := 0; j < n; j++ {
						g := out.grad[i*n+j]
						a.accumulate(i*k+p, g*b.data[p*n+j])
						b.accumulate(p*n+j, g*a.data[i*k+p])
					}
				}
			}
		}
	}
	return out
}

// Gather selects matrix rows, accumulating gradients for repeated indices.
func (a *Tensor) Gather(indices []int) *Tensor {
	if len(a.shape) != 2 || len(indices) == 0 {
		panic("gofish: Gather requires a matrix and nonempty indices")
	}
	indices = slices.Clone(indices)
	width := a.shape[1]
	data := make([]float64, len(indices)*width)
	for i, row := range indices {
		if row < 0 || row >= a.shape[0] {
			panic("gofish: embedding index out of range")
		}
		copy(data[i*width:], a.data[row*width:(row+1)*width])
	}
	out := node(data, []int{len(indices), width}, a)
	if out.requiresGrad {
		out.backward = func() {
			for i, row := range indices {
				for j := 0; j < width; j++ {
					a.accumulate(row*width+j, out.grad[i*width+j])
				}
			}
		}
	}
	return out
}

// Softmax normalizes each row along the last dimension, including vectors.
func (a *Tensor) Softmax() *Tensor {
	if len(a.shape) == 0 {
		panic("gofish: Softmax requires at least one dimension")
	}
	n := a.shape[len(a.shape)-1]
	data := make([]float64, len(a.data))
	for start := 0; start < len(data); start += n {
		peak := math.Inf(-1)
		for _, v := range a.data[start : start+n] {
			peak = max(peak, v)
		}
		sum := 0.0
		for j := 0; j < n; j++ {
			data[start+j] = math.Exp(a.data[start+j] - peak)
			sum += data[start+j]
		}
		for j := 0; j < n; j++ {
			data[start+j] /= sum
		}
	}
	out := node(data, a.shape, a)
	if out.requiresGrad {
		out.backward = func() {
			for start := 0; start < len(data); start += n {
				dot := 0.0
				for j := 0; j < n; j++ {
					dot += out.grad[start+j] * data[start+j]
				}
				for j := 0; j < n; j++ {
					a.accumulate(start+j, data[start+j]*(out.grad[start+j]-dot))
				}
			}
		}
	}
	return out
}

// CrossEntropy returns mean negative log likelihood from [batch, classes] logits.
// Log-sum-exp avoids overflow; targets contain one class index per row.
func CrossEntropy(logits *Tensor, targets []int) *Tensor {
	if len(logits.shape) != 2 || len(targets) != logits.shape[0] {
		panic("gofish: CrossEntropy shape mismatch")
	}
	b, n := logits.shape[0], logits.shape[1]
	targets = slices.Clone(targets)
	probs := make([]float64, len(logits.data))
	loss := 0.0
	for i, target := range targets {
		if target < 0 || target >= n {
			panic("gofish: target out of range")
		}
		row := logits.data[i*n : (i+1)*n]
		peak := math.Inf(-1)
		for _, v := range row {
			peak = max(peak, v)
		}
		sum := 0.0
		for j, v := range row {
			probs[i*n+j] = math.Exp(v - peak)
			sum += probs[i*n+j]
		}
		loss += (peak - row[target]) + math.Log(sum)
		for j := 0; j < n; j++ {
			probs[i*n+j] /= sum
		}
	}
	out := node([]float64{loss / float64(b)}, nil, logits)
	if out.requiresGrad {
		out.backward = func() {
			for i, target := range targets {
				for j := 0; j < n; j++ {
					g := probs[i*n+j]
					if j == target {
						g--
					}
					logits.accumulate(i*n+j, g*out.grad[0]/float64(b))
				}
			}
		}
	}
	return out
}

func MSE(prediction, target *Tensor) *Tensor { return prediction.Sub(target).Pow(2).Mean() }

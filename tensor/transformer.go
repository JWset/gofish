package tensor

import (
	"math"
	"slices"
)

// GELU applies the exact Gaussian error linear unit, x*Phi(x).
func (a *Tensor) GELU() *Tensor {
	return unary(a, func(x float64) float64 { return 0.5 * x * (1 + math.Erf(x/math.Sqrt2)) }, func(x float64) float64 {
		return 0.5*(1+math.Erf(x/math.Sqrt2)) + x*math.Exp(-0.5*x*x)/math.Sqrt(2*math.Pi)
	})
}

// LayerNorm normalizes the last dimension, using population variance and epsilon.
// Apply learned scale and bias with ordinary differentiable Mul and Add.
func (a *Tensor) LayerNorm(epsilon float64) *Tensor {
	if len(a.shape) == 0 || epsilon <= 0 || math.IsNaN(epsilon) || math.IsInf(epsilon, 0) {
		panic("gofish: LayerNorm requires a non-scalar and finite positive epsilon")
	}
	n := a.shape[len(a.shape)-1]
	data := make([]float64, len(a.data))
	inverse := make([]float64, len(a.data)/n)
	for row := range inverse {
		start := row * n
		mean := 0.0
		for _, v := range a.data[start : start+n] {
			mean += v
		}
		mean /= float64(n)
		variance := 0.0
		for _, v := range a.data[start : start+n] {
			d := v - mean
			variance += d * d
		}
		inverse[row] = 1 / math.Sqrt(variance/float64(n)+epsilon)
		for j := 0; j < n; j++ {
			data[start+j] = (a.data[start+j] - mean) * inverse[row]
		}
	}
	out := node(data, a.shape, a)
	if out.requiresGrad {
		out.backward = func() {
			for row, inv := range inverse {
				start := row * n
				sum, dot := 0.0, 0.0
				for j := 0; j < n; j++ {
					g := out.grad[start+j]
					sum += g
					dot += g * data[start+j]
				}
				for j := 0; j < n; j++ {
					a.accumulate(start+j, inv*(out.grad[start+j]-(sum+data[start+j]*dot)/float64(n)))
				}
			}
		}
	}
	return out
}

// Permute reorders all dimensions. The result owns its contiguous storage.
func (a *Tensor) Permute(axes ...int) *Tensor {
	if len(axes) != len(a.shape) {
		panic("gofish: Permute requires one axis per dimension")
	}
	seen := make([]bool, len(axes))
	shape := make([]int, len(axes))
	steps := strides(a.shape)
	for i, axis := range axes {
		if axis < 0 || axis >= len(axes) || seen[axis] {
			panic("gofish: invalid permutation")
		}
		seen[axis] = true
		shape[i] = a.shape[axis]
	}
	s := strides(shape)
	mapping := make([]int, len(a.data))
	data := make([]float64, len(a.data))
	for i := range data {
		for j, axis := range axes {
			mapping[i] += (i / s[j] % shape[j]) * steps[axis]
		}
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

// Slice selects a nonempty interval along one axis, copying its values.
func (a *Tensor) Slice(axis, start, end int) *Tensor {
	if axis < 0 {
		axis += len(a.shape)
	}
	if axis < 0 || axis >= len(a.shape) || start < 0 || end > a.shape[axis] || start >= end {
		panic("gofish: invalid tensor slice")
	}
	shape := slices.Clone(a.shape)
	shape[axis] = end - start
	s, original := strides(shape), strides(a.shape)
	mapping := make([]int, size(shape))
	data := make([]float64, len(mapping))
	for i := range data {
		for j := range shape {
			coordinate := i / s[j] % shape[j]
			if j == axis {
				coordinate += start
			}
			mapping[i] += coordinate * original[j]
		}
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

func batchMatMul(a, b *Tensor) *Tensor {
	if len(a.shape) < 2 || len(b.shape) < 2 || a.shape[len(a.shape)-1] != b.shape[len(b.shape)-2] {
		panic("gofish: MatMul requires compatible matrices")
	}
	batchShape := broadcast(a.shape[:len(a.shape)-2], b.shape[:len(b.shape)-2])
	m, k, n := a.shape[len(a.shape)-2], a.shape[len(a.shape)-1], b.shape[len(b.shape)-1]
	ai, bi := offsets(batchShape, a.shape[:len(a.shape)-2]), offsets(batchShape, b.shape[:len(b.shape)-2])
	shape := append(slices.Clone(batchShape), m, n)
	data := make([]float64, size(shape))
	for batch := range ai {
		aa, bb, cc := ai[batch]*m*k, bi[batch]*k*n, batch*m*n
		for i := 0; i < m; i++ {
			for p := 0; p < k; p++ {
				v := a.data[aa+i*k+p]
				for j := 0; j < n; j++ {
					data[cc+i*n+j] += v * b.data[bb+p*n+j]
				}
			}
		}
	}
	out := node(data, shape, a, b)
	if out.requiresGrad {
		out.backward = func() {
			for batch := range ai {
				aa, bb, cc := ai[batch]*m*k, bi[batch]*k*n, batch*m*n
				for i := 0; i < m; i++ {
					for p := 0; p < k; p++ {
						for j := 0; j < n; j++ {
							g := out.grad[cc+i*n+j]
							a.accumulate(aa+i*k+p, g*b.data[bb+p*n+j])
							b.accumulate(bb+p*n+j, g*a.data[aa+i*k+p])
						}
					}
				}
			}
		}
	}
	return out
}

// Attention computes multi-head scaled dot-product attention on [batch,time,width]
// tensors. It concatenates heads in the last dimension. Causal attention excludes
// every future key before softmax; query and key lengths must then match.
// This fused CPU implementation saves probabilities for exact reverse-mode AD.
func Attention(q, k, v *Tensor, heads int, causal bool) *Tensor {
	if len(q.shape) != 3 || len(k.shape) != 3 || len(v.shape) != 3 || !slices.Equal(k.shape, v.shape) || q.shape[0] != k.shape[0] || q.shape[2] != k.shape[2] || heads < 1 || q.shape[2]%heads != 0 || (causal && q.shape[1] != k.shape[1]) {
		panic("gofish: invalid multi-head attention shapes")
	}
	b, t, u, c := q.shape[0], q.shape[1], k.shape[1], q.shape[2]
	d := c / heads
	scale := 1 / math.Sqrt(float64(d))
	probabilities := make([]float64, size([]int{b, heads, t, u}))
	data := make([]float64, len(q.data))
	for batch := 0; batch < b; batch++ {
		for h := 0; h < heads; h++ {
			for i := 0; i < t; i++ {
				base := ((batch*heads+h)*t + i) * u
				qi := (batch*t+i)*c + h*d
				limit := u
				if causal {
					limit = i + 1
				}
				peak := math.Inf(-1)
				for j := 0; j < limit; j++ {
					kj := (batch*u+j)*c + h*d
					score := 0.0
					for p := 0; p < d; p++ {
						score += q.data[qi+p] * k.data[kj+p]
					}
					score *= scale
					probabilities[base+j] = score
					peak = max(peak, score)
				}
				sum := 0.0
				for j := 0; j < limit; j++ {
					probabilities[base+j] = math.Exp(probabilities[base+j] - peak)
					sum += probabilities[base+j]
				}
				for j := 0; j < limit; j++ {
					probabilities[base+j] /= sum
					vj := (batch*u+j)*c + h*d
					for p := 0; p < d; p++ {
						data[qi+p] += probabilities[base+j] * v.data[vj+p]
					}
				}
			}
		}
	}
	out := node(data, q.shape, q, k, v)
	if out.requiresGrad {
		out.backward = func() {
			dp := make([]float64, u)
			for batch := 0; batch < b; batch++ {
				for h := 0; h < heads; h++ {
					for i := 0; i < t; i++ {
						base := ((batch*heads+h)*t + i) * u
						qi := (batch*t+i)*c + h*d
						limit := u
						if causal {
							limit = i + 1
						}
						dot := 0.0
						for j := 0; j < limit; j++ {
							vj := (batch*u+j)*c + h*d
							g := 0.0
							for p := 0; p < d; p++ {
								g += out.grad[qi+p] * v.data[vj+p]
								v.accumulate(vj+p, probabilities[base+j]*out.grad[qi+p])
							}
							dp[j] = g
							dot += g * probabilities[base+j]
						}
						for j := 0; j < limit; j++ {
							kj := (batch*u+j)*c + h*d
							ds := probabilities[base+j] * (dp[j] - dot) * scale
							for p := 0; p < d; p++ {
								q.accumulate(qi+p, ds*k.data[kj+p])
								k.accumulate(kj+p, ds*q.data[qi+p])
							}
						}
					}
				}
			}
		}
	}
	return out
}

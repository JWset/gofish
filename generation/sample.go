package generation

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
)

// Sample applies temperature, top-k and nucleus (top-p) filtering to logits.
// allowed optionally excludes tokens before normalization. Logits are not changed.
func Sample(logits []float64, c Config, rng *rand.Rand, allowed func(int) bool) (int, error) {
	indices := make([]int, 0, len(logits))
	for i, v := range logits {
		if allowed != nil && !allowed(i) {
			continue
		}
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return 0, fmt.Errorf("gofish: non-finite generation logits")
		}
		indices = append(indices, i)
	}
	if len(indices) == 0 {
		return 0, fmt.Errorf("gofish: no allowed tokens")
	}
	sort.Slice(indices, func(i, j int) bool {
		a, b := indices[i], indices[j]
		if logits[a] == logits[b] {
			return a < b
		}
		return logits[a] > logits[b]
	})
	if c.Temperature == 0 {
		return indices[0], nil
	}
	if c.TopK > 0 && len(indices) > c.TopK {
		indices = indices[:c.TopK]
	}
	weights := make([]float64, len(indices))
	peak := logits[indices[0]]
	sum := 0.0
	for i, index := range indices {
		weights[i] = math.Exp((logits[index] - peak) / c.Temperature)
		sum += weights[i]
	}
	if c.TopP > 0 && c.TopP < 1 {
		cumulative := 0.0
		for i, w := range weights {
			cumulative += w
			if cumulative >= c.TopP*sum {
				indices = indices[:i+1]
				weights = weights[:i+1]
				sum = cumulative
				break
			}
		}
	}
	draw := rng.Float64() * sum
	for i, w := range weights {
		draw -= w
		if draw <= 0 {
			return indices[i], nil
		}
	}
	return indices[len(indices)-1], nil
}

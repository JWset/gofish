package optim

import (
	"fmt"
	"math"
	"slices"
)

type MomentState struct {
	First  []float64 `json:"first"`
	Second []float64 `json:"second"`
	Step   int       `json:"step"`
}

// AdamState is a copied, serializable optimizer state. Parameter order and sizes
// must match on restore. The state does not include model weights.
type AdamState struct {
	Version      int           `json:"version"`
	LearningRate float64       `json:"learning_rate"`
	WeightDecay  float64       `json:"weight_decay"`
	Moments      []MomentState `json:"moments"`
}

func (a *Adam) State() AdamState {
	s := AdamState{Version: 1, LearningRate: a.LearningRate, WeightDecay: a.WeightDecay}
	for _, m := range a.states {
		s.Moments = append(s.Moments, MomentState{First: slices.Clone(m.first), Second: slices.Clone(m.second), Step: m.step})
	}
	return s
}

// LoadState validates everything before replacing optimizer state.
func (a *Adam) LoadState(state AdamState) error {
	finite := func(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
	if state.Version != 1 || len(state.Moments) != len(a.states) || state.LearningRate <= 0 || !finite(state.LearningRate) || state.WeightDecay < 0 || !finite(state.WeightDecay) {
		return fmt.Errorf("gofish: incompatible Adam state")
	}
	for i, m := range state.Moments {
		if m.Step < 0 || len(m.First) != a.params[i].Numel() || len(m.Second) != a.params[i].Numel() {
			return fmt.Errorf("gofish: incompatible Adam moment %d", i)
		}
		for j, v := range m.First {
			if !finite(v) || !finite(m.Second[j]) || m.Second[j] < 0 {
				return fmt.Errorf("gofish: invalid Adam moment %d", i)
			}
		}
	}
	for i, m := range state.Moments {
		a.states[i] = adamState{first: slices.Clone(m.First), second: slices.Clone(m.Second), step: m.Step}
	}
	a.LearningRate = state.LearningRate
	a.WeightDecay = state.WeightDecay
	return nil
}

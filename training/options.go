// Package training contains shared native-model training options and progress.
package training

type Step struct {
	Number int
	Loss   float64
}
type Options struct {
	Steps        int
	BatchSize    int
	LearningRate float64
	ClipNorm     float64
	Seed         int64
	// Resume preserves transformer optimizer moments and update numbering. Each
	// sample batch is seeded from Seed and its absolute update number, allowing
	// exact continuation when the other options and corpus stay the same.
	// The character MLP backend does not support Resume.
	Resume      bool
	WeightDecay float64
	// OnStep runs synchronously with the model locked; do not re-enter the model.
	OnStep func(Step)
}
type History struct {
	Losses []float64
	Steps  int
}

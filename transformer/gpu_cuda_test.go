//go:build cuda

package transformer

import (
	"context"
	"math"
	"os"
	"testing"

	"github.com/JWset/gofish/training"
)

func TestCUDAUpdateMatchesCPU(t *testing.T) {
	if os.Getenv("GOFISH_CUDA_TEST") != "1" {
		t.Skip("set GOFISH_CUDA_TEST=1 to run on NVIDIA hardware")
	}
	cfg := Config{ContextSize: 4, Width: 8, Heads: 2, Layers: 1, FeedForward: 16, Seed: 42}
	cpu, _ := New(cfg)
	gpu, _ := New(cfg)
	options := training.Options{Steps: 1, BatchSize: 2, Seed: 7, WeightDecay: .01}
	corpus := "the fish swam in the sea."
	a, err := cpu.Train(context.Background(), corpus, options)
	if err != nil {
		t.Fatal(err)
	}
	options.Device = training.GPU
	b, err := gpu.Train(context.Background(), corpus, options)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(a.Losses[0]-b.Losses[0]) > 1e-5 {
		t.Fatalf("loss CPU=%v GPU=%v", a.Losses, b.Losses)
	}
	for i, p := range cpu.Parameters() {
		x, y := p.Data(), gpu.Parameters()[i].Data()
		for j := range x {
			if math.Abs(x[j]-y[j]) > 5e-5 {
				t.Fatalf("parameter %d/%d CPU=%g GPU=%g", i, j, x[j], y[j])
			}
		}
	}
	// CPU can resume the Adam state produced by CUDA and save/load it.
	path := t.TempDir() + "/cuda.gofish"
	if err = gpu.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	options.Device = training.CPU
	options.Resume = true
	if _, err = loaded.Train(context.Background(), corpus, options); err != nil {
		t.Fatal(err)
	}
	options.Device = training.GPU
	if _, err = gpu.Train(context.Background(), corpus, options); err != nil {
		t.Fatal(err)
	}
	if gpu.Updates() != 2 {
		t.Fatal(gpu.Updates())
	}
}

func TestCUDAConvergenceResumeAndCancellation(t *testing.T) {
	if os.Getenv("GOFISH_CUDA_TEST") != "1" {
		t.Skip("set GOFISH_CUDA_TEST=1 to run on NVIDIA hardware")
	}
	m, _ := New(Config{ContextSize: 4, Width: 8, Heads: 2, Layers: 1, FeedForward: 16, Seed: 42})
	corpus := "fish fish fish fish fish fish fish fish"
	initial, err := m.Loss(context.Background(), corpus)
	if err != nil {
		t.Fatal(err)
	}
	o := training.Options{Device: training.GPU, Steps: 60, BatchSize: 4, LearningRate: .01, Seed: 42}
	if _, err = m.Train(context.Background(), corpus, o); err != nil {
		t.Fatal(err)
	}
	final, err := m.Loss(context.Background(), corpus)
	if err != nil {
		t.Fatal(err)
	}
	if final >= initial-1 {
		t.Fatalf("GPU training did not learn: %g -> %g", initial, final)
	}
	t.Logf("GPU training corpus loss: %.4f -> %.4f", initial, final)
	path := t.TempDir() + "/gpu.gofish"
	if err = m.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	o.Steps = 2
	o.Resume = true
	if _, err = m.Train(context.Background(), corpus, o); err != nil {
		t.Fatal(err)
	}
	if _, err = loaded.Train(context.Background(), corpus, o); err != nil {
		t.Fatal(err)
	}
	for i, p := range m.Parameters() {
		a, b := p.Data(), loaded.Parameters()[i].Data()
		for j := range a {
			if math.Abs(a[j]-b[j]) > 1e-6 {
				t.Fatalf("GPU checkpoint resume differs at %d/%d", i, j)
			}
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	o.Steps = 5
	o.OnStep = func(s training.Step) {
		if s.Number == 2 {
			cancel()
		}
	}
	history, err := m.Train(ctx, corpus, o)
	if err != context.Canceled || history.Steps != 2 || m.Updates() != 64 {
		t.Fatalf("cancellation lost progress: %+v %v updates=%d", history, err, m.Updates())
	}
}

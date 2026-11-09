package nn_test

import (
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/JWset/gofish/nn"
	"github.com/JWset/gofish/optim"
	"github.com/JWset/gofish/tensor"
)

func TestXORTrainingAndCheckpoint(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	model := nn.NewSequential(nn.NewLinear(2, 8, rng), nn.Tanh{}, nn.NewLinear(8, 2, rng))
	x := tensor.New([]float64{0, 0, 0, 1, 1, 0, 1, 1}, 4, 2)
	targets := []int{0, 1, 1, 0}
	o := optim.NewAdam(model.Parameters(), 0.03)
	for step := 0; step < 300; step++ {
		o.ZeroGrad()
		loss := nn.CrossEntropy(model.Forward(x), targets)
		loss.Backward()
		o.Step()
	}
	prediction := model.Forward(x).Data()
	for i, want := range targets {
		got := 0
		if prediction[i*2+1] > prediction[i*2] {
			got = 1
		}
		if got != want {
			t.Fatalf("row %d: wrong XOR class", i)
		}
	}
	path := filepath.Join(t.TempDir(), "xor.json")
	if err := nn.Save(model, path); err != nil {
		t.Fatal(err)
	}
	clone := nn.NewSequential(nn.NewLinear(2, 8, rng), nn.Tanh{}, nn.NewLinear(8, 2, rng))
	if err := nn.Load(clone, path); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(prediction, clone.Forward(x).Data()) {
		t.Fatal("checkpoint changed predictions")
	}
	before := clone.Parameters()[0].Data()
	if err := os.WriteFile(path, []byte(`{"version":1,"weights":[{"shape":[999],"data":[1]}]}`), 0600); err != nil {
		t.Fatal(err)
	}
	if nn.Load(clone, path) == nil {
		t.Fatal("accepted incompatible checkpoint")
	}
	if !slices.Equal(before, clone.Parameters()[0].Data()) {
		t.Fatal("failed load mutated model")
	}
}

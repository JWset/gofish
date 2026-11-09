//go:build !cuda

package transformer

import (
	"context"
	"strings"
	"testing"

	"github.com/JWset/gofish/training"
)

func TestGPURequiresCUDABuild(t *testing.T) {
	m, _ := New(Config{Width: 8, Heads: 2, Layers: 1})
	before := m.Parameters()[0].Data()
	_, err := m.Train(context.Background(), "hello fish", training.Options{Device: training.GPU, Steps: 1})
	if err == nil || !strings.Contains(err.Error(), "-tags cuda") {
		t.Fatalf("got %v", err)
	}
	if m.Updates() != 0 || m.optimizer != nil {
		t.Fatal("failed GPU request changed training state")
	}
	after := m.Parameters()[0].Data()
	for i := range before {
		if before[i] != after[i] {
			t.Fatal("failed GPU request changed weights")
		}
	}
	_, err = m.Train(context.Background(), "hello fish", training.Options{Device: "gup", Steps: 1})
	if err == nil {
		t.Fatal("unknown device accepted")
	}
}

//go:build cuda

package transformer

import (
	"github.com/JWset/gofish/tensor"
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	_ "github.com/gomlx/compute/gobackend"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"math"
	"testing"
)

func TestAcceleratorGraphMatchesNative(t *testing.T) {
	backend, err := compute.NewWithConfig("go")
	if err != nil {
		t.Fatal(err)
	}
	defer backend.Finalize()
	m, _ := New(Config{ContextSize: 4, Width: 8, Heads: 2, Layers: 1, FeedForward: 16, Seed: 42})
	ids := []int{256, 102, 105, 115, 104, 32, 102, 105}
	in := []int32{256, 102, 105, 115, 104, 32, 102, 105}
	exec, err := graph.NewExec(backend, func(nodes []*graph.Node) []*graph.Node {
		logits := m.cudaForward(nodes[0], nodes[1:], 2, 4)
		targets := graph.Reshape(graph.Const(logits.Graph(), []int32{102, 105, 115, 104, 32, 102, 105, 115}), 2, 4)
		labels := graph.OneHot(targets, 257, dtypes.Float32)
		loss := graph.Neg(graph.ReduceAllMean(graph.ReduceSum(graph.Mul(graph.LogSoftmax(logits, -1), labels), -1)))
		return append([]*graph.Node{logits}, graph.Gradient(loss, nodes[1:]...)...)
	})
	if err != nil {
		t.Fatal(err)
	}
	defer exec.Finalize()
	buffers := []*tensors.Tensor{tensors.FromFlatDataAndDimensions(in, 2, 4)}
	defer func() { freeGPU(buffers) }()
	for _, p := range m.Parameters() {
		buffers = append(buffers, gpuTensor(p.Data(), p.Shape()))
	}
	args := make([]any, len(buffers))
	for i, p := range buffers {
		args[i] = p
	}
	outputs, err := exec.Call(args...)
	if err != nil {
		t.Fatal(err)
	}
	defer freeGPU(outputs)
	actual, err := gpuRead(outputs[0])
	if err != nil {
		t.Fatal(err)
	}
	native := m.Forward(ids, 2)
	tensor.CrossEntropy(native.Reshape(-1, 257), []int{102, 105, 115, 104, 32, 102, 105, 115}).Backward()
	want := native.Data()
	for i, p := range m.Parameters() {
		got, err := gpuRead(outputs[i+1])
		if err != nil {
			t.Fatal(err)
		}
		want := p.Grad()
		for j, g := range got {
			if math.Abs(g-want[j]) > 1e-5 {
				t.Fatalf("gradient %d/%d got=%g want=%g", i, j, g, want[j])
			}
		}
	}
	for i, v := range actual {
		if math.Abs(v-want[i]) > 1e-5 {
			t.Fatalf("logit %d got=%g want=%g", i, v, want[i])
		}
	}
}

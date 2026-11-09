//go:build cuda

package transformer

import (
	"context"
	"fmt"
	"math"
	"math/rand"

	"github.com/JWset/gofish/nn"
	"github.com/JWset/gofish/optim"
	"github.com/JWset/gofish/tensor"
	"github.com/JWset/gofish/tokenizer"
	"github.com/JWset/gofish/training"
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	_ "github.com/gomlx/go-xla/compute/xla/autoinstall"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
)

// cudaForward mirrors the CPU network, using float32 device tensors. Parameter
// identity preserves the native checkpoint layout and supports switching devices.
func (m *Model) cudaForward(in *Node, params []*Node, batch, length int) *Node {
	weights := map[*tensor.Tensor]*Node{}
	for i, p := range m.Parameters() {
		weights[p] = params[i]
	}
	linear := func(x *Node, l *nn.Linear) *Node {
		shape := x.Shape().Dimensions
		out := l.Bias.Numel()
		return Reshape(Add(DotProduct(Reshape(x, -1, l.Weight.Shape()[0]), weights[l.Weight]), Reshape(weights[l.Bias], 1, out)), shape[0], shape[1], out)
	}
	norm := func(x *Node, l *nn.LayerNorm) *Node {
		mean := Reshape(ReduceMean(x, -1), batch, length, 1)
		centered := Sub(x, mean)
		variance := Reshape(ReduceMean(Square(centered), -1), batch, length, 1)
		return Add(Mul(Div(centered, Sqrt(AddScalar(variance, l.Epsilon))), Reshape(weights[l.Weight], 1, 1, l.Weight.Numel())), Reshape(weights[l.Bias], 1, 1, l.Bias.Numel()))
	}
	c := m.config
	positionIDs := make([]int32, length)
	for i := range positionIDs {
		positionIDs[i] = int32(i)
	}
	pos := Gather(weights[m.net.positions.Weight], Reshape(Const(in.Graph(), positionIDs), length, 1))
	x := Add(Reshape(Gather(weights[m.net.tokens.Weight], Reshape(in, -1, 1)), batch, length, c.Width), Reshape(pos, 1, length, c.Width))
	mask := make([]float32, length*length)
	for i := 0; i < length; i++ {
		for j := i + 1; j < length; j++ {
			mask[i*length+j] = float32(math.Inf(-1))
		}
	}
	maskNode := Reshape(Const(in.Graph(), mask), 1, 1, length, length)
	for _, b := range m.net.blocks {
		h := norm(x, b.Norm1)
		a := b.Attention
		split := func(l *nn.Linear) *Node {
			return TransposeAllAxes(Reshape(linear(h, l), batch, length, c.Heads, c.Width/c.Heads), 0, 2, 1, 3)
		}
		q, k, v := split(a.Query), split(a.Key), split(a.Value)
		scores := MulScalar(DotGeneral(q, []int{3}, []int{0, 1}, k, []int{3}, []int{0, 1}), 1/math.Sqrt(float64(c.Width/c.Heads)))
		probs := Softmax(Add(scores, maskNode), -1)
		attended := DotGeneral(probs, []int{3}, []int{0, 1}, v, []int{2}, []int{0, 1})
		attended = Reshape(TransposeAllAxes(attended, 0, 2, 1, 3), batch, length, c.Width)
		x = Add(x, linear(attended, a.Projection))
		h = linear(norm(x, b.Norm2), b.FeedForward.Layers[0].(*nn.Linear))
		h = MulScalar(Mul(h, AddScalar(Erf(DivScalar(h, math.Sqrt2)), 1)), 0.5)
		x = Add(x, linear(h, b.FeedForward.Layers[2].(*nn.Linear)))
	}
	return linear(norm(x, m.net.norm), m.net.head)
}

func gpuTensor(data []float64, shape []int) *tensors.Tensor {
	f := make([]float32, len(data))
	for i, v := range data {
		f[i] = float32(v)
	}
	return tensors.FromFlatDataAndDimensions(f, shape...)
}
func gpuRead(t *tensors.Tensor) ([]float64, error) {
	f, err := tensors.CopyFlatData[float32](t)
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(f))
	for i, v := range f {
		out[i] = float64(v)
		if !finite(out[i]) {
			return nil, fmt.Errorf("gofish: non-finite GPU optimizer state")
		}
	}
	return out, nil
}
func freeGPU(ts []*tensors.Tensor) {
	for _, t := range ts {
		if t != nil {
			_ = t.FinalizeAll()
		}
	}
}

func (m *Model) trainGPU(ctx context.Context, ids []int, length int, o training.Options) (history training.History, err error) {
	if o.Resume && m.optimizer == nil {
		return history, fmt.Errorf("gofish: Resume requires an existing or loaded optimizer state")
	}
	// An explicit CUDA backend must fail if unavailable; never silently fall back.
	backend, err := compute.NewWithConfig("xla:cuda,preallocate=false")
	if err != nil {
		return history, fmt.Errorf("gofish: initialize CUDA training: %w", err)
	}
	defer backend.Finalize()
	if err = ctx.Err(); err != nil {
		return history, err
	}
	params := m.Parameters()
	count := len(params)
	optimizer := optim.NewAdamW(params, o.LearningRate, o.WeightDecay)
	if o.Resume {
		if m.optimizer == nil {
			return history, fmt.Errorf("gofish: Resume requires an existing or loaded optimizer state")
		}
		state := m.optimizer.State()
		state.LearningRate = o.LearningRate
		state.WeightDecay = o.WeightDecay
		if err = optimizer.LoadState(state); err != nil {
			return history, err
		}
	}
	state := optimizer.State()
	updates := 0
	if o.Resume {
		updates = m.updates
	}
	buffers := make([]*tensors.Tensor, 3*count)
	defer func() { freeGPU(buffers) }()
	for i, p := range params {
		buffers[i] = gpuTensor(p.Data(), p.Shape())
		buffers[count+i] = gpuTensor(state.Moments[i].First, p.Shape())
		buffers[2*count+i] = gpuTensor(state.Moments[i].Second, p.Shape())
	}
	exec, err := NewExec(backend, func(args []*Node) []*Node {
		weights := args[4 : 4+count]
		logits := m.cudaForward(args[0], weights, o.BatchSize, length)
		logProbs := LogSoftmax(logits, -1)
		labels := OneHot(args[1], tokenizer.VocabularySize, dtypes.Float32)
		loss := Neg(ReduceAllMean(ReduceSum(Mul(logProbs, labels), -1)))
		grads := Gradient(loss, weights...)
		normSquared := ReduceAllSum(Square(grads[0]))
		for _, g := range grads[1:] {
			normSquared = Add(normSquared, ReduceAllSum(Square(g)))
		}
		norm := Sqrt(normSquared)
		scale := MinScalar(Div(ConstAs(norm, o.ClipNorm), MaxScalar(norm, 1e-30)), 1)
		out := make([]*Node, 2+3*count)
		out[0] = loss
		out[1] = norm
		for i, w := range weights {
			g := Mul(grads[i], scale)
			first := Add(MulScalar(args[4+count+i], 0.9), MulScalar(g, 0.1))
			second := Add(MulScalar(args[4+2*count+i], 0.999), MulScalar(Square(g), 0.001))
			delta := Div(Div(first, args[2]), AddScalar(Sqrt(Div(second, args[3])), 1e-8))
			out[2+i] = Sub(MulScalar(w, 1-o.LearningRate*o.WeightDecay), MulScalar(delta, o.LearningRate))
			out[2+count+i] = first
			out[2+2*count+i] = second
		}
		return out
	})
	if err != nil {
		return history, err
	}
	defer exec.Finalize()
	// Copy back after completed updates, including cancellation. Stage all data
	// before changing native parameters, so a device read error cannot half-apply.
	defer func() {
		if history.Steps == 0 {
			return
		}
		data := make([][]float64, len(buffers))
		for i, t := range buffers {
			var e error
			data[i], e = gpuRead(t)
			if e != nil {
				err = fmt.Errorf("gofish: copy GPU training state: %w", e)
				return
			}
		}
		for i := range params {
			state.Moments[i] = optim.MomentState{First: data[count+i], Second: data[2*count+i], Step: state.Moments[i].Step + history.Steps}
		}
		if e := optimizer.LoadState(state); e != nil {
			err = e
			return
		}
		for i, p := range params {
			p.Assign(data[i])
		}
		m.optimizer = optimizer
		m.updates = updates
	}()
	for step := 0; step < o.Steps; step++ {
		if err = ctx.Err(); err != nil {
			return history, err
		}
		rng := rand.New(rand.NewSource(o.Seed ^ int64(updates)*-7046029254386353131))
		input, target := make([]int32, o.BatchSize*length), make([]int32, o.BatchSize*length)
		for row := 0; row < o.BatchSize; row++ {
			start := rng.Intn(len(ids) - length)
			for t := 0; t < length; t++ {
				input[row*length+t] = int32(ids[start+t])
				target[row*length+t] = int32(ids[start+t+1])
			}
		}
		inputs := tensors.FromFlatDataAndDimensions(input, o.BatchSize, length)
		targets := tensors.FromFlatDataAndDimensions(target, o.BatchSize, length)
		s := state.Moments[0].Step + step + 1
		c1, c2 := tensors.FromScalar(float32(1-math.Pow(.9, float64(s)))), tensors.FromScalar(float32(1-math.Pow(.999, float64(s))))
		args := []any{inputs, targets, c1, c2}
		for _, b := range buffers {
			args = append(args, b)
		}
		results, e := exec.Call(args...)
		freeGPU([]*tensors.Tensor{inputs, targets, c1, c2})
		if e != nil {
			return history, fmt.Errorf("gofish: CUDA training step: %w", e)
		}
		values, e := gpuRead(results[0])
		if e == nil {
			_, e = gpuRead(results[1])
		}
		if e != nil {
			freeGPU(results)
			return history, e
		}
		freeGPU(buffers)
		buffers = results[2:]
		freeGPU(results[:2])
		updates++
		history.Steps++
		history.Losses = append(history.Losses, values[0])
		if o.OnStep != nil {
			o.OnStep(training.Step{Number: step + 1, Loss: values[0]})
		}
	}
	return history, nil
}

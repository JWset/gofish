package optim_test

import (
	"math"
	"slices"
	"testing"

	"github.com/JWset/gofish/optim"
	"github.com/JWset/gofish/tensor"
)

func TestAdamWAndStateResume(t *testing.T) {
	p := tensor.New([]float64{2, -3}, 2).WithGrad()
	a := optim.NewAdamW([]*tensor.Tensor{p}, 0.01, 0.1)
	for i := 0; i < 3; i++ {
		a.ZeroGrad()
		p.Pow(2).Sum().Backward()
		a.Step()
	}
	q := p.Detach().WithGrad()
	b := optim.NewAdam([]*tensor.Tensor{q}, 0.5)
	state := a.State()
	if err := b.LoadState(state); err != nil {
		t.Fatal(err)
	}
	state.Moments[0].First[0] = 999
	for i := 0; i < 3; i++ {
		a.ZeroGrad()
		p.Pow(2).Sum().Backward()
		a.Step()
		b.ZeroGrad()
		q.Pow(2).Sum().Backward()
		b.Step()
	}
	if !slices.Equal(p.Data(), q.Data()) {
		t.Fatal("resumed updates differ", p.Data(), q.Data())
	}
	bad := a.State()
	bad.Moments[0].Second[0] = -1
	if b.LoadState(bad) == nil {
		t.Fatal("accepted negative second moment")
	}
	x := tensor.Scalar(2).WithGrad()
	x.Scale(0).Backward()
	optim.NewAdamW([]*tensor.Tensor{x}, 0.1, 0.2).Step()
	if math.Abs(x.Item()-1.96) > 1e-12 {
		t.Fatal("weight decay is not decoupled", x.Item())
	}
}

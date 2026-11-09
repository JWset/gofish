package optim_test

import (
	"math"
	"testing"

	"github.com/JWset/gofish/optim"
	"github.com/JWset/gofish/tensor"
)

func TestAdamBiasCorrection(t *testing.T) {
	p := tensor.Scalar(2).WithGrad()
	unused := tensor.Scalar(3).WithGrad()
	o := optim.NewAdam([]*tensor.Tensor{p, p, unused}, 0.1)
	for i := 0; i < 2; i++ {
		o.ZeroGrad()
		p.Scale(2).Backward()
		o.Step()
	}
	if math.Abs(p.Item()-1.8) > 1e-7 {
		t.Fatalf("first Adam updates: %g", p.Item())
	}
	if unused.Item() != 3 {
		t.Fatal("unused parameter updated")
	}
}
func TestClipAndSGD(t *testing.T) {
	p := tensor.New([]float64{3, 4}, 2).WithGrad()
	p.Pow(2).Sum().Backward()
	norm := optim.ClipGradNorm([]*tensor.Tensor{p, p}, 5)
	if norm != 10 {
		t.Fatal(norm)
	}
	optim.NewSGD([]*tensor.Tensor{p}, 0.1).Step()
	data := p.Data()
	if math.Abs(data[0]-2.7) > 1e-10 || math.Abs(data[1]-3.6) > 1e-10 {
		t.Fatal(data)
	}
}

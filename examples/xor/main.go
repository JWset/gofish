package main

import (
	"fmt"
	"math/rand"

	"github.com/JWset/gofish"
	"github.com/JWset/gofish/nn"
	"github.com/JWset/gofish/optim"
)

func main() {
	rng := rand.New(rand.NewSource(42))
	model := nn.NewSequential(nn.NewLinear(2, 8, rng), nn.Tanh{}, nn.NewLinear(8, 2, rng))
	x := gofish.NewTensor([]float64{0, 0, 0, 1, 1, 0, 1, 1}, 4, 2)
	labels := []int{0, 1, 1, 0}
	optimizer := optim.NewAdam(model.Parameters(), 0.03)
	for step := 0; step < 300; step++ {
		optimizer.ZeroGrad()
		loss := nn.CrossEntropy(model.Forward(x), labels)
		loss.Backward()
		optimizer.Step()
		if step%100 == 0 {
			fmt.Printf("step %d loss %.4f\n", step, loss.Item())
		}
	}
	probabilities := model.Forward(x).Softmax().Data()
	for i := range labels {
		fmt.Printf("%v -> P(1)=%.4f\n", x.Data()[i*2:i*2+2], probabilities[i*2+1])
	}
}

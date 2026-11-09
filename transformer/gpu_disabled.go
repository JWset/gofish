//go:build !cuda

package transformer

import (
	"context"
	"fmt"
	"github.com/JWset/gofish/training"
)

func (m *Model) trainGPU(_ context.Context, _ []int, _ int, _ training.Options) (training.History, error) {
	return training.History{}, fmt.Errorf("gofish: GPU training requires a CUDA build; run with go run -tags cuda . (see GPU.md)")
}

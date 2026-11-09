package gofish_test

import (
	"context"
	"errors"
	"path/filepath"
	"sync"
	"testing"

	"github.com/JWset/gofish"
)

func TestConcurrentNativeOperations(t *testing.T) {
	m, err := gofish.Load("tiny-stories")
	if err != nil {
		t.Fatal(err)
	}
	var group sync.WaitGroup
	for i := 0; i < 4; i++ {
		group.Add(1)
		go func() {
			defer group.Done()
			if _, err := m.Generate(context.Background(), "the sun was", gofish.WithMaxTokens(20)); err != nil {
				t.Error(err)
			}
		}()
	}
	group.Add(1)
	go func() {
		defer group.Done()
		if _, err := m.Train(context.Background(), "the sun was warm.\n", gofish.TrainOptions{Steps: 2}); err != nil {
			t.Error(err)
		}
	}()
	group.Wait()
}

func TestPublicAPI(t *testing.T) {
	ctx := context.Background()
	m, err := gofish.Load("tiny-dialogue")
	if err != nil {
		t.Fatal(err)
	}
	if !m.Capabilities().Train || m.Capabilities().Pull {
		t.Fatal("incorrect native capabilities")
	}
	if _, err = m.Train(ctx, "user: hello\nfish: hello!\n", gofish.TrainOptions{Steps: 1}); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "fine-tuned.gofish")
	if err = m.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := gofish.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = loaded.Generate(ctx, "user: hello\nfish:", gofish.WithMaxTokens(20)); err != nil {
		t.Fatal(err)
	}
	if err = m.Pull(ctx); !errors.Is(err, gofish.ErrUnsupported) {
		t.Fatal(err)
	}
	remote, err := gofish.Load("ollama:qwen3:0.6b")
	if err != nil {
		t.Fatal(err)
	}
	if !remote.Capabilities().Chat || remote.Capabilities().Train {
		t.Fatal("incorrect Ollama capabilities")
	}
	if _, err = remote.Train(ctx, "text", gofish.TrainOptions{}); !errors.Is(err, gofish.ErrUnsupported) {
		t.Fatal(err)
	}
	if err = remote.Save(path); !errors.Is(err, gofish.ErrUnsupported) {
		t.Fatal(err)
	}
	if _, err = gofish.Load("ollama:"); err == nil {
		t.Fatal("accepted empty remote model")
	}
	if _, err = gofish.Load(filepath.Join(t.TempDir(), "missing")); err == nil {
		t.Fatal("accepted missing checkpoint")
	}
	x := gofish.NewTensor([]float64{2}, 1).WithGrad()
	x.Pow(2).Sum().Backward()
	if x.Grad()[0] != 4 {
		t.Fatal("root tensor API broken")
	}
}

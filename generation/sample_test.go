package generation_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/JWset/gofish/generation"
)

func TestSamplingFilters(t *testing.T) {
	for _, option := range []generation.Option{generation.WithTemperature(0), generation.WithTopK(1), generation.WithTopP(0.01)} {
		c, err := generation.Resolve([]generation.Option{option})
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < 20; i++ {
			id, err := generation.Sample([]float64{1, 3, 2}, c, rand.New(rand.NewSource(int64(i))), nil)
			if err != nil || id != 1 {
				t.Fatal(id, err)
			}
		}
	}
	c, _ := generation.Resolve(nil)
	id, err := generation.Sample([]float64{1, 3, 2}, c, rand.New(rand.NewSource(1)), func(i int) bool { return i == 0 })
	if err != nil || id != 0 {
		t.Fatal(id, err)
	}
	if _, err = generation.Sample([]float64{math.NaN()}, c, rand.New(rand.NewSource(1)), nil); err == nil {
		t.Fatal("accepted NaN logits")
	}
	if _, err = generation.Sample([]float64{1}, c, rand.New(rand.NewSource(1)), func(int) bool { return false }); err == nil {
		t.Fatal("accepted empty allowed set")
	}
	for _, option := range []generation.Option{generation.WithTopK(-1), generation.WithTopP(0), generation.WithTopP(2), generation.WithTopP(math.NaN())} {
		if _, err = generation.Resolve([]generation.Option{option}); err == nil {
			t.Fatal("accepted invalid filtering option")
		}
	}
}

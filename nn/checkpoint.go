package nn

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
)

type weight struct {
	Shape []int     `json:"shape"`
	Data  []float64 `json:"data"`
}
type checkpoint struct {
	Version int      `json:"version"`
	Weights []weight `json:"weights"`
}

// Save writes module weights atomically. Recreate the same architecture before Load.
func Save(module Module, path string) error {
	c := checkpoint{Version: 1}
	for _, p := range module.Parameters() {
		c.Weights = append(c.Weights, weight{p.Shape(), p.Data()})
	}
	data, err := json.Marshal(c)
	if err != nil {
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".gofish-*")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	if _, err = f.Write(data); err != nil {
		f.Close()
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	return os.Rename(f.Name(), path)
}

// Load validates all tensors before changing any parameter.
func Load(module Module, path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var c checkpoint
	if err = json.Unmarshal(data, &c); err != nil {
		return err
	}
	params := module.Parameters()
	if c.Version != 1 || len(c.Weights) != len(params) {
		return fmt.Errorf("gofish: incompatible checkpoint")
	}
	for i, p := range params {
		w := c.Weights[i]
		if !slices.Equal(w.Shape, p.Shape()) || len(w.Data) != p.Numel() {
			return fmt.Errorf("gofish: incompatible parameter %d", i)
		}
		for _, v := range w.Data {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return fmt.Errorf("gofish: non-finite parameter %d", i)
			}
		}
	}
	for i, p := range params {
		p.Assign(c.Weights[i].Data)
	}
	return nil
}

package transformer

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/JWset/gofish/optim"
)

const maxCheckpointBytes = 256 << 20

type checkpoint struct {
	Format    string           `json:"format"`
	Version   int              `json:"version"`
	Config    Config           `json:"config"`
	Weights   [][]float64      `json:"weights"`
	Updates   int              `json:"updates"`
	Optimizer *optim.AdamState `json:"optimizer,omitempty"`
}

// Save atomically writes architecture, trained weights and any AdamW state.
func (m *Model) Save(path string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	c := checkpoint{Format: "gofish-byte-transformer", Version: 1, Config: m.config, Updates: m.updates}
	for _, p := range m.Parameters() {
		c.Weights = append(c.Weights, p.Data())
	}
	if m.optimizer != nil {
		s := m.optimizer.State()
		c.Optimizer = &s
	}
	data, err := json.Marshal(c)
	if err != nil {
		return err
	}
	if len(data) > maxCheckpointBytes {
		return fmt.Errorf("gofish: transformer checkpoint exceeds decoded size limit")
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".gofish-*")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	z := gzip.NewWriter(f)
	if _, err = z.Write(data); err != nil {
		z.Close()
		f.Close()
		return err
	}
	if err = z.Close(); err != nil {
		f.Close()
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	return os.Rename(f.Name(), path)
}
func decode(reader io.Reader) (*Model, error) {
	z, err := gzip.NewReader(reader)
	if err != nil {
		return nil, fmt.Errorf("gofish: read transformer checkpoint: %w", err)
	}
	defer z.Close()
	data, err := io.ReadAll(io.LimitReader(z, maxCheckpointBytes+1))
	if err != nil {
		return nil, err
	}
	if len(data) > maxCheckpointBytes {
		return nil, fmt.Errorf("gofish: transformer checkpoint exceeds decoded size limit")
	}
	var c checkpoint
	if err = json.Unmarshal(data, &c); err != nil {
		return nil, fmt.Errorf("gofish: decode transformer checkpoint: %w", err)
	}
	if c.Format != "gofish-byte-transformer" || c.Version != 1 {
		return nil, fmt.Errorf("gofish: unsupported transformer checkpoint format")
	}
	if err = validate(c.Config); err != nil {
		return nil, err
	}
	if c.Updates < 0 {
		return nil, fmt.Errorf("gofish: invalid transformer update count")
	}
	count := 0
	for _, w := range c.Weights {
		count += len(w)
		for _, v := range w {
			if !finite(v) {
				return nil, fmt.Errorf("gofish: non-finite transformer checkpoint weights")
			}
		}
	}
	if count != parameterCount(c.Config) {
		return nil, fmt.Errorf("gofish: transformer parameter count mismatch")
	}
	m, err := New(c.Config)
	if err != nil {
		return nil, err
	}
	params := m.Parameters()
	if len(params) != len(c.Weights) {
		return nil, fmt.Errorf("gofish: transformer parameter layout mismatch")
	}
	for i, p := range params {
		if len(c.Weights[i]) != p.Numel() {
			return nil, fmt.Errorf("gofish: transformer parameter %d size mismatch", i)
		}
	}
	if c.Optimizer != nil {
		m.optimizer = optim.NewAdam(params, 0.001)
		if err = m.optimizer.LoadState(*c.Optimizer); err != nil {
			return nil, err
		}
		for _, state := range c.Optimizer.Moments {
			if state.Step != c.Updates {
				return nil, fmt.Errorf("gofish: optimizer step count differs from model updates")
			}
		}
	} else if c.Updates != 0 {
		return nil, fmt.Errorf("gofish: trained checkpoint is missing optimizer state")
	}
	for i, p := range params {
		p.Assign(c.Weights[i])
	}
	m.updates = c.Updates
	return m, nil
}
func Load(path string) (*Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return decode(f)
}
func LoadBytes(data []byte) (*Model, error) { return decode(bytes.NewReader(data)) }

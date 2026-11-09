package lm

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"unicode/utf8"
)

const maxCheckpointBytes = 64 << 20

type checkpoint struct {
	Format     string      `json:"format"`
	Version    int         `json:"version"`
	Config     Config      `json:"config"`
	Vocabulary string      `json:"vocabulary"`
	Weights    [][]float64 `json:"weights"`
}

// Save writes a gzip-compressed JSON checkpoint atomically. It contains architecture,
// vocabulary and weights; optimizer moments are deliberately not included.
func (m *Model) Save(path string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	c := checkpoint{Format: "gofish-char-mlp", Version: 1, Config: m.config, Vocabulary: string(m.vocabulary)}
	for _, p := range m.Parameters() {
		c.Weights = append(c.Weights, p.Data())
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".gofish-*")
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	z := gzip.NewWriter(f)
	if err = json.NewEncoder(z).Encode(c); err != nil {
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
		return nil, fmt.Errorf("gofish: read checkpoint: %w", err)
	}
	defer z.Close()
	data, err := io.ReadAll(io.LimitReader(z, maxCheckpointBytes+1))
	if err != nil {
		return nil, err
	}
	if len(data) > maxCheckpointBytes {
		return nil, fmt.Errorf("gofish: checkpoint exceeds 64 MiB decoded limit")
	}
	var c checkpoint
	if err = json.Unmarshal(data, &c); err != nil {
		return nil, fmt.Errorf("gofish: decode checkpoint: %w", err)
	}
	if c.Format != "gofish-char-mlp" || c.Version != 1 {
		return nil, fmt.Errorf("gofish: unsupported checkpoint format or version")
	}
	if !utf8.ValidString(c.Vocabulary) {
		return nil, fmt.Errorf("gofish: invalid vocabulary encoding")
	}
	vocab := []rune(c.Vocabulary)
	if len(vocab) < 2 || vocab[0] != 0 {
		return nil, fmt.Errorf("gofish: invalid padding token")
	}
	seen := map[rune]bool{}
	for _, r := range vocab {
		if seen[r] {
			return nil, fmt.Errorf("gofish: duplicate vocabulary character")
		}
		seen[r] = true
	}
	// Validate dimensions and weight lengths before allocating model tensors.
	if err = validate(c.Config, len(vocab)); err != nil {
		return nil, err
	}
	lengths := []int{len(vocab) * c.Config.EmbeddingSize, c.Config.ContextSize * c.Config.EmbeddingSize * c.Config.HiddenSize, c.Config.HiddenSize, c.Config.HiddenSize * len(vocab), len(vocab)}
	if len(c.Weights) != len(lengths) {
		return nil, fmt.Errorf("gofish: invalid parameter count")
	}
	for i, n := range lengths {
		if len(c.Weights[i]) != n {
			return nil, fmt.Errorf("gofish: invalid parameter %d size", i)
		}
		for _, v := range c.Weights[i] {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return nil, fmt.Errorf("gofish: non-finite checkpoint weights")
			}
		}
	}
	m, err := newModel(vocab, c.Config)
	if err != nil {
		return nil, err
	}
	for i, p := range m.Parameters() {
		p.Assign(c.Weights[i])
	}
	return m, nil
}

// Load loads a native checkpoint saved with Save. For bundled models use Pretrained.
func Load(path string) (*Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return decode(f)
}

// LoadBytes loads compressed checkpoint bytes, including embedded application assets.
func LoadBytes(data []byte) (*Model, error) { return decode(bytes.NewReader(data)) }

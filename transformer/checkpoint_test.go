package transformer_test

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/JWset/gofish/training"
	"github.com/JWset/gofish/transformer"
)

func TestRejectCheckpointCorruption(t *testing.T) {
	m := newModel(t)
	if _, err := m.Train(context.Background(), "hello fish", training.Options{Steps: 1, BatchSize: 1}); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "model.gofish")
	if err := m.Save(path); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name string
		edit func(map[string]any)
	}{
		{"format", func(c map[string]any) { c["format"] = "unknown" }},
		{"version", func(c map[string]any) { c["version"] = 99 }},
		{"config", func(c map[string]any) { c["config"].(map[string]any)["width"] = 0 }},
		{"weights", func(c map[string]any) { c["weights"] = []any{} }},
		{"missing optimizer", func(c map[string]any) { delete(c, "optimizer") }},
		{"updates", func(c map[string]any) { c["updates"] = -1 }},
		{"moments", func(c map[string]any) { c["optimizer"].(map[string]any)["moments"] = []any{} }},
	} {
		t.Run(test.name, func(t *testing.T) {
			z, err := gzip.NewReader(bytes.NewReader(data))
			if err != nil {
				t.Fatal(err)
			}
			var fields map[string]any
			if err = json.NewDecoder(z).Decode(&fields); err != nil {
				t.Fatal(err)
			}
			z.Close()
			test.edit(fields)
			var buffer bytes.Buffer
			w := gzip.NewWriter(&buffer)
			if err = json.NewEncoder(w).Encode(fields); err != nil {
				t.Fatal(err)
			}
			if err = w.Close(); err != nil {
				t.Fatal(err)
			}
			if _, err = transformer.LoadBytes(buffer.Bytes()); err == nil {
				t.Fatal("accepted corrupt checkpoint")
			}
		})
	}
}

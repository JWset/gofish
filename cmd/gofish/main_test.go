package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCLIGenerateTrainAndLoad(t *testing.T) {
	var output, diagnostics bytes.Buffer
	if err := run(context.Background(), []string{"generate", "-prompt", "the sun was", "-temperature", "0", "-tokens", "30"}, strings.NewReader(""), &output, &diagnostics); err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(output.String(), " warm and the river was quiet.") {
		t.Fatal(output.String())
	}
	dir := t.TempDir()
	corpus := filepath.Join(dir, "corpus.txt")
	checkpoint := filepath.Join(dir, "model.gofish")
	if err := os.WriteFile(corpus, []byte("hello fish. hello bird.\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := run(context.Background(), []string{"train", "-model", "new", "-data", corpus, "-out", checkpoint, "-steps", "20"}, strings.NewReader(""), &output, &diagnostics); err != nil {
		t.Fatal(err)
	}
	if err := run(context.Background(), []string{"generate", "-model", checkpoint, "-prompt", "hello", "-tokens", "10"}, strings.NewReader(""), &output, &diagnostics); err != nil {
		t.Fatal(err)
	}
}
func TestCLIUsageAndErrors(t *testing.T) {
	for _, args := range [][]string{nil, {"models"}, {"generate", "-help"}} {
		if err := run(context.Background(), args, strings.NewReader(""), &bytes.Buffer{}, &bytes.Buffer{}); err != nil {
			t.Fatal(err)
		}
	}
	for _, args := range [][]string{{"invalid"}, {"train", "-model", "new"}, {"chat"}, {"generate", "-timeout", "0"}, {"generate", "extra"}} {
		if err := run(context.Background(), args, strings.NewReader(""), &bytes.Buffer{}, &bytes.Buffer{}); err == nil {
			t.Fatalf("accepted %v", args)
		}
	}
}

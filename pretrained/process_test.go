package pretrained

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestStartupCancellationAndRuntimeErrors(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX process fixture")
	}
	path := metadataFile(t, "qwen3", "gpt2", nil)
	dir := t.TempDir()
	script := filepath.Join(dir, "runtime")
	if err := os.WriteFile(script, []byte("#!/bin/sh\nexec sleep 60\n"), 0700); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	start := time.Now()
	_, err := Load(ctx, path, Options{ServerPath: script})
	if !errors.Is(err, context.DeadlineExceeded) || time.Since(start) > 2*time.Second {
		t.Fatalf("startup cancellation: %v", err)
	}
	if err = os.WriteFile(script, []byte("#!/bin/sh\necho 'incompatible tokenizer fixture' >&2\nexit 2\n"), 0700); err != nil {
		t.Fatal(err)
	}
	_, err = Load(context.Background(), path, Options{ServerPath: script})
	if err == nil || !strings.Contains(err.Error(), "incompatible tokenizer fixture") {
		t.Fatalf("lost runtime error: %v", err)
	}
	_, err = Load(context.Background(), path, Options{ServerPath: filepath.Join(dir, "missing")})
	if err == nil || !strings.Contains(err.Error(), "requires llama-server") {
		t.Fatal(err)
	}
}

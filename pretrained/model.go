// Package pretrained runs explicitly supported GGUF families through a managed,
// loopback-only llama.cpp server. Weights and conversations stay local.
package pretrained

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/JWset/gofish/gguf"
)

var ErrArchitecture = errors.New("gofish: unsupported GGUF architecture")
var ErrClosed = errors.New("gofish: pretrained model is closed")

type Options struct {
	ContextSize    int
	MaxModelBytes  uint64
	MaxCacheBytes  uint64
	ServerPath     string
	GPULayers      int // Zero selects CPU, positive values offload layers to a GPU build of llama-server.
	Threads        int
	ChatTemplate   string // Path to an explicit Jinja template; otherwise GGUF metadata is used.
	StartupTimeout time.Duration
}
type Info struct {
	TokenizerAddsSpacePrefix bool
	Architecture             string
	Family                   string
	Tokenizer                string
	Name                     string
	ContextSize              int
	DeclaredContext          int
	ParameterCount           int
	HasChatTemplate          bool
	Quantizations            []string
}

// Inspect checks architecture/tokenizer combinations without loading tensor payloads.
func Inspect(path string, options Options) (Info, error) {
	var info Info
	f, err := gguf.Open(path)
	if err != nil {
		return info, err
	}
	defer f.Close()
	if f.Uint("split.count", 1) > 1 {
		return info, fmt.Errorf("gofish: split GGUF files are not supported; merge shards first")
	}
	info.Architecture = f.String("general.architecture", "")
	info.Tokenizer = f.String("tokenizer.ggml.model", "")
	info.Name = f.String("general.name", "")
	info.TokenizerAddsSpacePrefix = f.Bool("tokenizer.ggml.add_space_prefix", info.Tokenizer == "llama")
	tokenizers := map[string]string{"qwen3": "gpt2", "gemma": "llama", "gemma2": "llama", "gemma3": "llama", "smollm3": "gpt2", "llama": ""}
	want, ok := tokenizers[info.Architecture]
	if !ok {
		return info, fmt.Errorf("%w %q; supported: qwen3 (dense), gemma, gemma2, gemma3 (text), llama/SmolLM1/2, smollm3", ErrArchitecture, info.Architecture)
	}
	if want != "" && info.Tokenizer != want {
		return info, fmt.Errorf("gofish: architecture %s requires tokenizer %s, got %q", info.Architecture, want, info.Tokenizer)
	}
	if info.Architecture == "llama" && info.Tokenizer != "gpt2" && info.Tokenizer != "llama" {
		return info, fmt.Errorf("gofish: unsupported Llama tokenizer %q", info.Tokenizer)
	}
	if !f.Bool(info.Architecture+".attention.causal", true) {
		return info, fmt.Errorf("%w: noncausal/embedding model %s", ErrArchitecture, info.Architecture)
	}
	if f.Uint(info.Architecture+".expert_count", 0) > 0 {
		return info, fmt.Errorf("%w: MoE variant of %s", ErrArchitecture, info.Architecture)
	}
	info.Family = info.Architecture
	if info.Architecture == "llama" && (strings.HasPrefix(f.String("tokenizer.ggml.pre", ""), "smollm") || strings.Contains(strings.ToLower(info.Name), "smollm")) {
		info.Family = "smollm"
	}
	declared := f.Uint(info.Architecture+".context_length", 0)
	if declared < 1 || declared > 1<<24 {
		return info, fmt.Errorf("gofish: invalid GGUF context length %d", declared)
	}
	info.DeclaredContext = int(declared)
	info.ContextSize = options.ContextSize
	if info.ContextSize == 0 {
		info.ContextSize = min(2048, info.DeclaredContext)
	}
	if info.ContextSize < 1 || info.ContextSize > info.DeclaredContext {
		return info, fmt.Errorf("gofish: requested context %d exceeds declared context %d or is invalid", info.ContextSize, info.DeclaredContext)
	}
	if options.GPULayers < 0 || options.Threads < 0 || options.StartupTimeout < 0 {
		return info, fmt.Errorf("gofish: invalid pretrained runtime options")
	}
	limit := options.MaxModelBytes
	if limit == 0 {
		limit = 16 << 30
	}
	stat, err := os.Stat(path)
	if err != nil {
		return info, err
	}
	if uint64(stat.Size()) > limit {
		return info, fmt.Errorf("gofish: GGUF file exceeds MaxModelBytes (%d)", limit)
	}
	cache := options.MaxCacheBytes
	if cache == 0 {
		cache = 1 << 30
	}
	prefix := info.Architecture + "."
	width, heads, layers := f.Uint(prefix+"embedding_length", 0), f.Uint(prefix+"attention.head_count", 0), f.Uint(prefix+"block_count", 0)
	kv := f.Uint(prefix+"attention.head_count_kv", heads)
	if width == 0 || heads == 0 || layers == 0 || width > 1<<20 || heads > 65536 || layers > 4096 || kv == 0 || kv > heads {
		return info, fmt.Errorf("gofish: invalid pretrained model dimensions")
	}
	key := f.Uint(prefix+"attention.key_length", width/heads)
	value := f.Uint(prefix+"attention.value_length", key)
	if key > 1<<20 || value > 1<<20 || uint64(info.ContextSize) > cache/max(uint64(1), 2*layers*kv*(key+value)) {
		return info, fmt.Errorf("gofish: requested context exceeds estimated FP16 KV cache budget (%d bytes); reduce ContextSize or increase MaxCacheBytes", cache)
	}
	info.HasChatTemplate = f.String("tokenizer.chat_template", "") != "" || options.ChatTemplate != ""
	kinds := map[string]bool{}
	for _, t := range f.Tensors {
		n := t.Elements()
		if n > 1<<40 || uint64(info.ParameterCount)+n > uint64(^uint(0)>>1) {
			return info, fmt.Errorf("gofish: excessive tensor dimensions")
		}
		info.ParameterCount += int(n)
		k := t.Type.String()
		if !kinds[k] {
			kinds[k] = true
			info.Quantizations = append(info.Quantizations, k)
		}
	}
	return info, nil
}

type tailLog struct {
	sync.Mutex
	data []byte
}

func (l *tailLog) Write(p []byte) (int, error) {
	l.Lock()
	defer l.Unlock()
	l.data = append(l.data, p...)
	if len(l.data) > 8192 {
		l.data = l.data[len(l.data)-8192:]
	}
	return len(p), nil
}
func (l *tailLog) String() string { l.Lock(); defer l.Unlock(); return string(l.data) }

type Model struct {
	info     Info
	url, key string
	client   *http.Client
	cmd      *exec.Cmd
	done     chan struct{}
	waitErr  error
	log      tailLog
	ctx      context.Context
	cancel   context.CancelFunc
	once     sync.Once
}

func (m *Model) Info() Info {
	info := m.info
	info.Quantizations = append([]string(nil), info.Quantizations...)
	return info
}

func Load(ctx context.Context, path string, options Options) (*Model, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	info, err := Inspect(path, options)
	if err != nil {
		return nil, err
	}
	binary := options.ServerPath
	if binary == "" {
		binary = os.Getenv("GOFISH_LLAMA_SERVER")
	}
	if binary == "" {
		binary = "llama-server"
	}
	binary, err = exec.LookPath(binary)
	if err != nil {
		return nil, fmt.Errorf("gofish: local pretrained inference requires llama-server (tested with llama.cpp b12242); install it or set GGUFOptions.ServerPath / GOFISH_LLAMA_SERVER: %w", err)
	}
	path, err = filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	listener, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		return nil, err
	}
	port := listener.Addr().(*net.TCPAddr).Port
	listener.Close()
	secret := make([]byte, 32)
	if _, err = rand.Read(secret); err != nil {
		return nil, err
	}
	m := &Model{info: info, key: hex.EncodeToString(secret), url: fmt.Sprintf("http://127.0.0.1:%d", port), done: make(chan struct{}), client: &http.Client{Transport: &http.Transport{Proxy: nil, DisableKeepAlives: true}, CheckRedirect: func(*http.Request, []*http.Request) error {
		return fmt.Errorf("gofish: local runtime redirects are disabled")
	}}}
	m.ctx, m.cancel = context.WithCancel(context.Background())
	args := []string{"--model", path, "--host", "127.0.0.1", "--port", fmt.Sprint(port), "--api-key", m.key, "--ctx-size", fmt.Sprint(info.ContextSize), "--parallel", "1", "--n-gpu-layers", fmt.Sprint(options.GPULayers), "--alias", "gofish-local", "--jinja", "--offline", "--no-warmup", "--no-context-shift"}
	if options.Threads > 0 {
		args = append(args, "--threads", fmt.Sprint(options.Threads))
	}
	if options.ChatTemplate != "" {
		args = append(args, "--chat-template-file", options.ChatTemplate)
	}
	m.cmd = exec.Command(binary, args...)
	m.cmd.Stdout = &m.log
	m.cmd.Stderr = &m.log
	if err = m.cmd.Start(); err != nil {
		m.cancel()
		return nil, fmt.Errorf("gofish: start local runtime: %w", err)
	}
	go func() { m.waitErr = m.cmd.Wait(); close(m.done) }()
	timeout := options.StartupTimeout
	if timeout == 0 {
		timeout = 2 * time.Minute
	}
	startup, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-startup.Done():
			m.Close()
			return nil, fmt.Errorf("gofish: loading %s: %w; runtime: %s", info.Architecture, startup.Err(), m.log.String())
		case <-m.done:
			m.Close()
			return nil, fmt.Errorf("gofish: runtime rejected %s GGUF: %v; %s", info.Architecture, m.waitErr, m.log.String())
		case <-ticker.C:
			request, _ := http.NewRequestWithContext(startup, "GET", m.url+"/health", nil)
			request.Header.Set("Authorization", "Bearer "+m.key)
			response, e := m.client.Do(request)
			if e == nil {
				response.Body.Close()
				if response.StatusCode == http.StatusOK {
					return m, nil
				}
			}
		}
	}
}
func (m *Model) Close() error {
	m.once.Do(func() {
		m.cancel()
		if m.cmd != nil && m.cmd.Process != nil {
			_ = m.cmd.Process.Kill()
			<-m.done
		}
		m.client.CloseIdleConnections()
	})
	return nil
}

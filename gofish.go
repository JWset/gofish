// Package gofish combines tensors, autograd, CPU/CUDA transformer training, and
// local pretrained GGUF inference with an optional Ollama adapter.
package gofish

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/JWset/gofish/chat"
	"github.com/JWset/gofish/generation"
	"github.com/JWset/gofish/gguf"
	"github.com/JWset/gofish/internal/checkpoint"
	"github.com/JWset/gofish/llama"
	"github.com/JWset/gofish/lm"
	"github.com/JWset/gofish/ollama"
	"github.com/JWset/gofish/pretrained"
	"github.com/JWset/gofish/tensor"
	"github.com/JWset/gofish/training"
	"github.com/JWset/gofish/transformer"
)

type Tensor = tensor.Tensor

var NewTensor = tensor.New
var Scalar = tensor.Scalar
var Zeros = tensor.Zeros
var Ones = tensor.Ones
var Randn = tensor.Randn

type Config = lm.Config
type TransformerConfig = transformer.Config
type LlamaConfig = llama.Config
type GGUFOptions = llama.LoadOptions
type Device = training.Device

const (
	CPU = training.CPU
	GPU = training.GPU
)

type TrainOptions = lm.TrainOptions
type History = lm.History
type Step = lm.Step
type GenerateOption = generation.Option
type Message = ollama.Message
type Tool = chat.Tool
type Function = chat.Function
type ToolCall = chat.ToolCall
type FunctionCall = chat.FunctionCall
type ChatOptions = chat.Options
type ChatResponse = chat.Response
type ChatDelta = chat.Delta
type PretrainedInfo = pretrained.Info

var WithMaxTokens = generation.WithMaxTokens
var WithTemperature = generation.WithTemperature
var WithSeed = generation.WithSeed
var WithTopK = generation.WithTopK
var WithTopP = generation.WithTopP
var WithStop = generation.WithStop
var WithStream = generation.WithStream

var ErrUnsupportedArchitecture = pretrained.ErrArchitecture

var ErrUnsupported = errors.New("gofish: operation is unsupported by this model backend")

// Model wraps a native trainable model, a local pretrained GGUF runtime, or Ollama.
// Native premade models can be fine-tuned; Ollama models support inference only.
type nativeModel interface {
	Generate(context.Context, string, ...generation.Option) (string, error)
	Train(context.Context, string, TrainOptions) (History, error)
	Loss(context.Context, string) (float64, error)
	Save(string) error
	Parameters() []*tensor.Tensor
}
type Model struct {
	native  nativeModel
	remote  *ollama.Model
	weights *llama.Model
	local   *pretrained.Model
}
type Capabilities struct {
	Train bool
	Save  bool
	Pull  bool
	Chat  bool
}

func (m *Model) Capabilities() Capabilities {
	if m.local != nil {
		return Capabilities{Chat: true}
	}
	if m.weights != nil {
		return Capabilities{Train: m.weights.CanTrain(), Save: m.weights.CanTrain()}
	}
	return Capabilities{Train: m.native != nil, Save: m.native != nil, Pull: m.remote != nil, Chat: m.remote != nil}
}

// Load accepts "tiny-stories", "tiny-dialogue", "tiny-transformer", a native checkpoint path, or
// "ollama:<model-name>". Ollama weights must be downloaded separately with Pull.
func Load(nameOrPath string) (*Model, error) {
	if strings.HasPrefix(nameOrPath, "ollama:") {
		return Ollama(strings.TrimPrefix(nameOrPath, "ollama:"))
	}
	if nameOrPath == "tiny" {
		nameOrPath = "tiny-stories"
	}
	if nameOrPath == "tiny-llama" {
		m, err := llama.Pretrained()
		if err != nil {
			return nil, err
		}
		return &Model{weights: m}, nil
	}
	if nameOrPath == "tiny-transformer" {
		m, err := transformer.Pretrained()
		if err != nil {
			return nil, err
		}
		return &Model{native: m}, nil
	}
	if nameOrPath == "tiny-stories" || nameOrPath == "tiny-dialogue" {
		m, err := lm.Pretrained(nameOrPath)
		if err != nil {
			return nil, err
		}
		return &Model{native: m}, nil
	}
	f, err := os.Open(nameOrPath)
	if err != nil {
		return nil, err
	}
	var magic [4]byte
	n, _ := f.Read(magic[:])
	f.Close()
	if n == 4 && string(magic[:]) == "GGUF" {
		return LoadGGUF(nameOrPath)
	}
	format, err := checkpoint.Format(nameOrPath)
	if err != nil {
		return nil, err
	}
	var model nativeModel
	switch format {
	case "gofish-char-mlp":
		model, err = lm.Load(nameOrPath)
	case "gofish-byte-transformer":
		model, err = transformer.Load(nameOrPath)
	default:
		return nil, fmt.Errorf("gofish: unsupported checkpoint format %q", format)
	}
	if err != nil {
		return nil, err
	}
	return &Model{native: model}, nil
}

// LoadGGUF preserves the native fixture backend and dispatches pretrained architectures
// to a managed local llama.cpp runtime. Runtime:"native" explicitly selects the native subset.
func LoadGGUF(path string, options ...GGUFOptions) (*Model, error) {
	return LoadGGUFContext(context.Background(), path, options...)
}

// LoadGGUFContext is LoadGGUF with cancellation during local-runtime startup.
func LoadGGUFContext(ctx context.Context, path string, options ...GGUFOptions) (*Model, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(options) > 1 {
		return nil, fmt.Errorf("gofish: LoadGGUF accepts at most one options struct")
	}
	var config GGUFOptions
	if len(options) == 1 {
		config = options[0]
	}
	if config.Runtime != "" && config.Runtime != "auto" && config.Runtime != "native" && config.Runtime != "llama.cpp" {
		return nil, fmt.Errorf("gofish: Runtime must be auto, native, or llama.cpp")
	}
	useLocal := config.Runtime == "llama.cpp"
	if config.Runtime == "" || config.Runtime == "auto" {
		f, err := gguf.Open(path)
		if err != nil {
			return nil, err
		}
		arch, tok := f.String("general.architecture", ""), f.String("tokenizer.ggml.model", "")
		useLocal = arch != "llama" || (tok != "llama" && tok != "gofish-byte") || f.String("tokenizer.chat_template", "") != ""
		f.Close()
	}
	if useLocal {
		return LoadPretrained(ctx, path, config)
	}

	m, err := llama.LoadWithOptions(path, config)
	if err != nil {
		return nil, err
	}
	return &Model{weights: m}, nil
}
func NewLlama(config LlamaConfig) (*Model, error) {
	m, err := llama.NewByteModel(config)
	if err != nil {
		return nil, err
	}
	return &Model{weights: m}, nil
}

// NewTransformer initializes a byte-tokenized decoder for native CPU/CUDA training.
func NewTransformer(config TransformerConfig) (*Model, error) {
	m, err := transformer.New(config)
	if err != nil {
		return nil, err
	}
	return &Model{native: m}, nil
}

// ParameterCount reports native parameter count; Ollama models are external and return zero.
func (m *Model) ParameterCount() int {
	if m.local != nil {
		return m.local.Info().ParameterCount
	}
	if m.weights != nil {
		return m.weights.ParameterCount()
	}
	if m.native == nil {
		return 0
	}
	n := 0
	for _, p := range m.native.Parameters() {
		n += p.Numel()
	}
	return n
}
func NewLanguageModel(corpus string, config Config) (*Model, error) {
	m, err := lm.New(corpus, config)
	if err != nil {
		return nil, err
	}
	return &Model{native: m}, nil
}
func Ollama(name string, options ...ollama.Option) (*Model, error) {
	m, err := ollama.New(name, options...)
	if err != nil {
		return nil, err
	}
	return &Model{remote: m}, nil
}
func (m *Model) Generate(ctx context.Context, prompt string, options ...GenerateOption) (string, error) {
	if m.local != nil {
		return m.local.Generate(ctx, prompt, options...)
	}
	if m.weights != nil {
		return m.weights.Generate(ctx, prompt, options...)
	}
	if m.native != nil {
		return m.native.Generate(ctx, prompt, options...)
	}
	if m.remote != nil {
		return m.remote.Generate(ctx, prompt, options...)
	}
	return "", fmt.Errorf("gofish: uninitialized model")
}
func (m *Model) Train(ctx context.Context, text string, options TrainOptions) (History, error) {
	if m.local != nil {
		return History{}, fmt.Errorf("%w: managed pretrained GGUF models support inference only", ErrUnsupported)
	}
	if m.weights != nil {
		if !m.weights.CanTrain() {
			return History{}, fmt.Errorf("%w: native GGUF training is limited to 2 million parameters", ErrUnsupported)
		}
		return m.weights.Train(ctx, text, options)
	}
	if m.native == nil {
		return History{}, fmt.Errorf("%w: training requires a native model; Ollama fine-tuning is not implemented", ErrUnsupported)
	}
	return m.native.Train(ctx, text, options)
}
func (m *Model) Save(path string) error {
	if m.local != nil {
		return fmt.Errorf("%w: managed pretrained GGUF models use their original weight file", ErrUnsupported)
	}
	if m.weights != nil {
		if !m.weights.CanTrain() {
			return fmt.Errorf("%w: GGUF export is limited to 2 million parameters", ErrUnsupported)
		}
		return m.weights.Save(path)
	}
	if m.native == nil {
		return fmt.Errorf("%w: Save requires a native model", ErrUnsupported)
	}
	return m.native.Save(path)
}
func (m *Model) Pull(ctx context.Context) error {
	if m.remote == nil {
		return fmt.Errorf("%w: bundled native models need no download", ErrUnsupported)
	}
	return m.remote.Pull(ctx)
}
func (m *Model) Chat(ctx context.Context, messages []Message, options ...GenerateOption) (string, error) {
	if m.local != nil {
		return m.local.Chat(ctx, messages, options...)
	}
	if m.remote == nil {
		return "", fmt.Errorf("%w: chat requires a pretrained GGUF or Ollama model; native demo models generate continuations", ErrUnsupported)
	}
	return m.remote.Chat(ctx, messages, options...)
}
func (m *Model) Loss(ctx context.Context, text string) (float64, error) {
	if m.weights != nil {
		return m.weights.Loss(ctx, text)
	}
	if m.native == nil {
		return 0, fmt.Errorf("%w: Loss requires a native model", ErrUnsupported)
	}
	return m.native.Loss(ctx, text)
}

// Close releases the managed local runtime. It is safe to call more than once.
func (m *Model) Close() error {
	if m.local != nil {
		return m.local.Close()
	}
	return nil
}
func (m *Model) PretrainedInfo() (PretrainedInfo, error) {
	if m.local != nil {
		return m.local.Info(), nil
	}
	return PretrainedInfo{}, ErrUnsupported
}

// ChatStructured returns text and typed function calls; it never executes functions.
func (m *Model) ChatStructured(ctx context.Context, messages []Message, config ChatOptions, options ...GenerateOption) (ChatResponse, error) {
	if m.local != nil {
		return m.local.ChatStructured(ctx, messages, config, options...)
	}
	return ChatResponse{}, fmt.Errorf("%w: structured chat requires a local pretrained GGUF model", ErrUnsupported)
}
func (m *Model) Tokenize(ctx context.Context, text string) ([]int, error) {
	if m.local != nil {
		return m.local.Tokenize(ctx, text)
	}
	return nil, ErrUnsupported
}
func (m *Model) Detokenize(ctx context.Context, tokens []int) (string, error) {
	if m.local != nil {
		return m.local.Detokenize(ctx, tokens)
	}
	return "", ErrUnsupported
}
func (m *Model) ApplyChatTemplate(ctx context.Context, messages []Message) (string, error) {
	if m.local != nil {
		return m.local.ApplyChatTemplate(ctx, messages)
	}
	return "", ErrUnsupported
}

// LoadPretrained loads a supported GGUF into a managed local runtime with cancellable startup.
func LoadPretrained(ctx context.Context, path string, options ...GGUFOptions) (*Model, error) {
	if len(options) > 1 {
		return nil, fmt.Errorf("gofish: LoadPretrained accepts at most one options struct")
	}
	var c GGUFOptions
	if len(options) == 1 {
		c = options[0]
	}
	if c.Runtime != "" && c.Runtime != "auto" && c.Runtime != "llama.cpp" {
		return nil, fmt.Errorf("gofish: LoadPretrained requires the llama.cpp runtime")
	}
	model, err := pretrained.Load(ctx, path, pretrained.Options{ContextSize: c.ContextSize, MaxModelBytes: c.MaxModelBytes, MaxCacheBytes: c.MaxCacheBytes, ServerPath: c.ServerPath, GPULayers: c.GPULayers, Threads: c.Threads, ChatTemplate: c.ChatTemplate, StartupTimeout: c.StartupTimeout})
	if err != nil {
		return nil, err
	}
	return &Model{local: model}, nil
}

// PretrainedModels lists reference GGUF checkpoints with pinned upstream revisions.
func PretrainedModels() []pretrained.Preset { return pretrained.Models() }

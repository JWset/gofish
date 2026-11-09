// Package gofish combines PyTorch-inspired tensors, autograd and neural network
// training with a small language-model API. Native models are CPU-only float64;
// larger pretrained models can run through a separately installed Ollama server.
package gofish

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/JWset/gofish/generation"
	"github.com/JWset/gofish/internal/checkpoint"
	"github.com/JWset/gofish/llama"
	"github.com/JWset/gofish/lm"
	"github.com/JWset/gofish/ollama"
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

var WithMaxTokens = generation.WithMaxTokens
var WithTemperature = generation.WithTemperature
var WithSeed = generation.WithSeed
var WithTopK = generation.WithTopK
var WithTopP = generation.WithTopP
var WithStop = generation.WithStop
var WithStream = generation.WithStream

var ErrUnsupported = errors.New("gofish: operation is unsupported by this model backend")

// Model wraps a native character model, native byte transformer, or Ollama client.
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
}
type Capabilities struct {
	Train bool
	Save  bool
	Pull  bool
	Chat  bool
}

func (m *Model) Capabilities() Capabilities {
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

// LoadGGUF loads the supported Llama GGUF subset directly, without an external runtime.
func LoadGGUF(path string, options ...GGUFOptions) (*Model, error) {
	if len(options) > 1 {
		return nil, fmt.Errorf("gofish: LoadGGUF accepts at most one options struct")
	}
	var config GGUFOptions
	if len(options) == 1 {
		config = options[0]
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

// NewTransformer initializes a byte-tokenized decoder for native CPU training.
func NewTransformer(config TransformerConfig) (*Model, error) {
	m, err := transformer.New(config)
	if err != nil {
		return nil, err
	}
	return &Model{native: m}, nil
}

// ParameterCount reports native parameter count; Ollama models are external and return zero.
func (m *Model) ParameterCount() int {
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
	if m.remote == nil {
		return "", fmt.Errorf("%w: chat requires an Ollama model; native models generate continuations", ErrUnsupported)
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

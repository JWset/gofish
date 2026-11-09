// Package gofish combines PyTorch-inspired tensors, autograd and neural network
// training with a small language-model API. Native models are CPU-only float64;
// larger pretrained models can run through a separately installed Ollama server.
package gofish

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/JWset/gofish/generation"
	"github.com/JWset/gofish/lm"
	"github.com/JWset/gofish/ollama"
	"github.com/JWset/gofish/tensor"
)

type Tensor = tensor.Tensor

var NewTensor = tensor.New
var Scalar = tensor.Scalar
var Zeros = tensor.Zeros
var Ones = tensor.Ones
var Randn = tensor.Randn

type Config = lm.Config
type TrainOptions = lm.TrainOptions
type History = lm.History
type Step = lm.Step
type GenerateOption = generation.Option
type Message = ollama.Message

var WithMaxTokens = generation.WithMaxTokens
var WithTemperature = generation.WithTemperature
var WithSeed = generation.WithSeed
var WithStop = generation.WithStop
var WithStream = generation.WithStream

var ErrUnsupported = errors.New("gofish: operation is unsupported by this model backend")

// Model wraps either a native trainable character model or an Ollama client.
// Native premade models can be fine-tuned; Ollama models support inference only.
type Model struct {
	native *lm.Model
	remote *ollama.Model
}
type Capabilities struct {
	Train bool
	Save  bool
	Pull  bool
	Chat  bool
}

func (m *Model) Capabilities() Capabilities {
	return Capabilities{Train: m.native != nil, Save: m.native != nil, Pull: m.remote != nil, Chat: m.remote != nil}
}

// Load accepts "tiny-stories", "tiny-dialogue", a native checkpoint path, or
// "ollama:<model-name>". Ollama weights must be downloaded separately with Pull.
func Load(nameOrPath string) (*Model, error) {
	if strings.HasPrefix(nameOrPath, "ollama:") {
		return Ollama(strings.TrimPrefix(nameOrPath, "ollama:"))
	}
	if nameOrPath == "tiny" {
		nameOrPath = "tiny-stories"
	}
	if nameOrPath == "tiny-stories" || nameOrPath == "tiny-dialogue" {
		m, err := lm.Pretrained(nameOrPath)
		if err != nil {
			return nil, err
		}
		return &Model{native: m}, nil
	}
	m, err := lm.Load(nameOrPath)
	if err != nil {
		return nil, err
	}
	return &Model{native: m}, nil
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
	if m.native != nil {
		return m.native.Generate(ctx, prompt, options...)
	}
	if m.remote != nil {
		return m.remote.Generate(ctx, prompt, options...)
	}
	return "", fmt.Errorf("gofish: uninitialized model")
}
func (m *Model) Train(ctx context.Context, text string, options TrainOptions) (History, error) {
	if m.native == nil {
		return History{}, fmt.Errorf("%w: training requires a native model; Ollama fine-tuning is not implemented", ErrUnsupported)
	}
	return m.native.Train(ctx, text, options)
}
func (m *Model) Save(path string) error {
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
	if m.native == nil {
		return 0, fmt.Errorf("%w: Loss requires a native model", ErrUnsupported)
	}
	return m.native.Loss(ctx, text)
}

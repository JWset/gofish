# GoFish

Tensors, autograd, neural networks, and text generation in Go. No Python, CGO,
API key, or third-party Go dependencies are needed for the native backend.

**This is a small, working first version inspired by PyTorch, not a complete
PyTorch replacement.** Native training uses CPU float64 tensors. It includes two
small pretrained character models and an Ollama adapter for running larger
pretrained instruction models. Ollama models are inference-only in GoFish.

## Try it now

From this repository, with Go 1.22 or later:

```sh
go run ./examples/tiny
go run ./examples/xor
go run ./cmd/gofish models
go run ./cmd/gofish generate -prompt "the sun was" -temperature 0
```

The tiny model runs immediately, with no downloads. Its weights are embedded in
the module. It generates continuations of its small training corpus; it is not
a general-purpose chatbot. See [MODEL_CARD.md](MODEL_CARD.md).

## Import it

The intended public module path is:

```go
import "github.com/JWset/gofish"
```

Once the repository is published, install with:

```sh
go get github.com/JWset/gofish
go install github.com/JWset/gofish/cmd/gofish@latest
```

Before publication, use it from another local Go module with:

```sh
go mod edit -require=github.com/JWset/gofish@v0.0.0
go mod edit -replace=github.com/JWset/gofish=/absolute/path/to/gofish
go mod tidy
```

## A premade model in Go

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/JWset/gofish"
)

func main() {
    model, err := gofish.Load("tiny-stories")
    if err != nil {
        log.Fatal(err)
    }
    text, err := model.Generate(context.Background(), "the sun was",
        gofish.WithMaxTokens(100),
        gofish.WithTemperature(0),
    )
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("the sun was" + text)
}
```

`Generate` returns the continuation, excluding the prompt. For native models,
tokens are Unicode characters; prompts and training text must use the model's
existing vocabulary. Errors report unfamiliar characters. Create a new model
from your corpus when you need a different vocabulary.

`WithTemperature(0)` enables greedy decoding. `WithSeed` makes sampling
reproducible. `WithStop` removes matching stop sequences, including across
streamed chunks. `WithStream(func(string) error)` receives generated chunks;
return an error to stop. Callbacks run synchronously and must not re-enter the
same native model.

## Run a larger pretrained model

Install [Ollama](https://ollama.com/download) and start its server (`ollama serve`)
if it is not already running. Then download a model explicitly:

```sh
go run ./cmd/gofish pull -model ollama:qwen3:0.6b
go run ./cmd/gofish chat -model ollama:qwen3:0.6b
```

Downloads can be hundreds of megabytes or several gigabytes, depending on the
model. GoFish never downloads weights merely because you call `Load`.

The same works in code:

```go
model, err := gofish.Load("ollama:qwen3:0.6b")
if err != nil {
    return err
}
if err := model.Pull(ctx); err != nil {
    return err
}
reply, err := model.Chat(ctx, []gofish.Message{
    {Role: "system", Content: "Be helpful and concise."},
    {Role: "user", Content: "Tell me about the ocean."},
}, gofish.WithMaxTokens(200))
```

Pass a context deadline for requests and downloads. Conversation history is
explicit: append the assistant's reply and your next user message when calling
`Chat` again. Any model installed in Ollama can be selected with the `ollama:`
prefix; actual capabilities, hardware requirements, and licenses depend on that
model. The starter catalog lists Qwen3 0.6B and 4B. See the
[Qwen3 model page](https://ollama.com/library/qwen3:0.6b) and
[Ollama API](https://docs.ollama.com/api/chat).

For another server or custom HTTP client:

```go
model, err := gofish.Ollama("qwen3:0.6b",
    ollama.WithBaseURL("http://localhost:11434"),
    ollama.WithHTTPClient(&http.Client{Timeout: 5 * time.Minute}),
)
```

Import `github.com/JWset/gofish/ollama` for these options.
There is no hosted token charge when using locally hosted weights. You supply
the hardware and electricity. Remote servers may have their own charges.

## Train and save your own model

Create an untrained character model with a vocabulary derived from your text:

```go
model, err := gofish.NewLanguageModel(corpus, gofish.Config{
    ContextSize:   12,
    EmbeddingSize: 8,
    HiddenSize:    64,
    Seed:          42,
})
if err != nil {
    return err
}
history, err := model.Train(ctx, corpus, gofish.TrainOptions{
    Steps:        1000,
    BatchSize:    32,
    LearningRate: 0.01,
    Seed:         42,
})
if err != nil {
    return err
}
fmt.Println(history.Losses[len(history.Losses)-1])
if err := model.Save("my-model.gofish"); err != nil {
    return err
}
```

Reload with `gofish.Load("my-model.gofish")`, then call `Generate` or `Train`.
Both bundled native pretrained models can also be trained further this way.
Each `Train` call starts fresh Adam moments; checkpoints save the architecture,
vocabulary, and weights, not optimizer state. Steps count updates, not epochs.

CLI equivalents:

```sh
# Train from scratch using the vocabulary in your text.
go run ./cmd/gofish train -model new -data my-text.txt -steps 1000 -out my-model.gofish
go run ./cmd/gofish generate -model my-model.gofish -prompt "hello"

# Continue training a bundled model on compatible text.
go run ./cmd/gofish train -model tiny-stories -data lm/corpora/tiny-stories.txt -steps 100 -out tuned.gofish
```

Native models use a fixed-context embedding/MLP architecture, not a transformer.
They are suitable for small experiments. GoFish cannot load Hugging Face,
PyTorch, GGUF, or ONNX weights into its native training backend. Native model
dimensions are bounded to keep these experiments small (at most 5 million
parameters). Fine-tuning Qwen/Llama and GPU training are not implemented.
Check `model.Capabilities()` before backend-specific operations; unsupported
operations return errors wrapping `gofish.ErrUnsupported`.

## Build neural networks without implementing neurons

```go
rng := rand.New(rand.NewSource(42))
model := nn.NewSequential(
    nn.NewLinear(2, 8, rng),
    nn.Tanh{},
    nn.NewLinear(8, 2, rng),
)
optimizer := optim.NewAdam(model.Parameters(), 0.03)

for step := 0; step < 300; step++ {
    optimizer.ZeroGrad()
    prediction := model.Forward(inputs)
    loss := nn.CrossEntropy(prediction, labels)
    loss.Backward()
    optimizer.Step()
}
```

See [examples/xor](examples/xor/main.go) for a complete program. `nn.Save` and
`nn.Load` save and restore weights for a recreated network architecture. The
checkpoint validates all shapes before applying any updates.

Available packages:

| Package | Provides |
| --- | --- |
| `gofish` | Model loading, training, generation, starter catalog, tensor constructors |
| `tensor` | Dense float64 tensors, broadcasting, matrix multiply, reshape, transpose, row gather, reductions, activations, softmax, losses, autograd |
| `nn` | Linear, Embedding, Sequential, ReLU, Tanh, Sigmoid, network weight checkpoints |
| `optim` | SGD, bias-corrected Adam, global gradient norm clipping |
| `lm` | Native character-model architecture, vocabulary, training and compressed checkpoints |
| `ollama` | Pretrained generation, explicit chat history, streaming and model downloads |
| `generation` | Shared sampling options and stop-aware stream collection |

## Tensor behavior

```go
x := gofish.NewTensor([]float64{2, 3}, 2).WithGrad()
loss := x.Mul(x).Sum()
loss.Backward()
fmt.Println(x.Grad()) // [4 6]
```

Tensors own their storage; `Data`, `Shape`, and `Grad` return copies. Arithmetic
supports NumPy-style broadcasting. `MatMul` and `Transpose` currently require
2D matrices. `Softmax` normalizes the last dimension. `CrossEntropy` takes
`[batch, classes]` logits and one integer target per row.

`Backward` accumulates leaf gradients, so call `ZeroGrad` before the next update.
Non-scalar outputs need a same-shaped upstream gradient passed to `Backward`.
Graphs can be reused before parameters change; modifying a value needed by a
pending backward pass produces a panic. `Detach` copies values without their
graph. Shape/index errors panic, while model loading, generation and training
return errors. Higher-order derivatives, sparse tensors, GPU tensors, batched
matrix multiplication, convolution, and distributed training are not included.

Individual tensors are not safe for concurrent mutation. High-level native
`Train`, `Generate`, `Loss`, and `Save` calls serialize access to a model; direct
parameter mutation requires your own synchronization.

## Verify and reproduce

```sh
go test -race -cover ./...
go vet ./...
go run ./internal/pretrain
```

Tests check derivatives with finite differences, shared computation graphs,
broadcasting, embedding gradients, numerical stability, XOR convergence,
language-model convergence, actual bundled checkpoints, save/load parity,
Unicode, cancellation, streaming stops, CLI training, and Ollama request/response
contracts. Adapter tests use in-process HTTP transports and do not download or
execute a real instruction model.

The pretraining command regenerates both bundled checkpoints from the included
original corpora. There are no external training datasets. MIT licensed.

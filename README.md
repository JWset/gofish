# GoFish

Tensors, autograd, neural networks, and text generation in Go. No Python, CGO,
API key, or third-party Go dependencies are needed for the native backend.

**This is a small, working first version inspired by PyTorch, not a complete
PyTorch replacement.** Native training uses CPU float64 tensors. It includes a
byte-level decoder transformer, native GGUF Llama inference, two character models, and an Ollama adapter for running larger
pretrained instruction models. Ollama models are inference-only in GoFish.

## Try it now

From this repository, with Go 1.22 or later:

```sh
go run ./examples/tiny
go run ./examples/transformer
go run ./examples/gguf
go run ./examples/xor
go run ./cmd/gofish models
go run ./cmd/gofish generate -prompt "the sun was" -temperature 0
```

All four tiny models run immediately, with no downloads. Their trained weights
are embedded in the module. They generate patterns from small training corpora;
they are not general-purpose chatbots. See [MODEL_CARD.md](MODEL_CARD.md).

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

`Generate` returns the continuation, excluding the prompt. `tiny-stories` and
`tiny-dialogue` use Unicode character tokens and a fixed vocabulary. Their errors
report unfamiliar characters. `tiny-transformer` uses byte tokens and can accept
any valid UTF-8 prompt or training text, including characters absent from training.

`WithTemperature(0)` enables greedy decoding. `WithSeed` makes sampling
reproducible. `WithTopK(k)` and `WithTopP(p)` control top-k and nucleus sampling.
Top-k zero disables the native filter; on Ollama it leaves the server default.
`WithStop` removes matching stop sequences, including across
streamed chunks. `WithStream(func(string) error)` receives generated chunks;
return an error to stop. Callbacks run synchronously and must not re-enter the
same native model.

## A native transformer

Load a trained transformer using the same API:

```go
model, err := gofish.Load("tiny-transformer")
if err != nil {
    return err
}
text, err := model.Generate(ctx, "the little fish",
    gofish.WithMaxTokens(160),
    gofish.WithTemperature(0.5),
    gofish.WithTopK(20),
    gofish.WithTopP(0.9),
)
```

This checkpoint has 34,881 parameters, two decoder blocks, four attention heads,
32-wide embeddings, and a 32-byte context window. It implements causal attention,
learned positions, pre-layer normalization, residual connections, and GELU feed-forward
layers. It was trained on the included original corpus. It is still a tiny text
generation demonstration, not a general assistant.

Or create your own transformer, train it, and save a resumable checkpoint:

```go
model, err := gofish.NewTransformer(gofish.TransformerConfig{
    ContextSize: 32,
    Width:       32,
    Heads:       4,
    Layers:      2,
    FeedForward: 64,
    Seed:        42,
})
if err != nil {
    return err
}
_, err = model.Train(ctx, corpus, gofish.TrainOptions{
    Steps:        1000,
    BatchSize:    4,
    LearningRate: 0.003,
    WeightDecay:  0.01,
    Seed:         42,
})
if err != nil {
    return err
}
if err := model.Save("transformer.gofish"); err != nil {
    return err
}
```

Continue training with optimizer moments restored:

```go
model, err := gofish.Load("transformer.gofish")
if err != nil {
    return err
}
_, err = model.Train(ctx, corpus, gofish.TrainOptions{
    Steps:     500,
    BatchSize: 4,
    Seed:      42,
    Resume:    true,
})
```

Transformer checkpoints include weights, architecture, AdamW moments and absolute
update numbering. `Resume: true` preserves that state; an ordinary `Train` call
starts fresh optimizer state while retaining the model's weights. To reproduce an
uninterrupted run exactly, keep the corpus, seed, batch size and other training
settings the same. A zero learning rate or weight decay on resume retains the
saved optimizer value; start a fresh optimizer to reset decay to zero.

CLI equivalents:

```sh
go run ./cmd/gofish generate -model tiny-transformer -prompt "the little fish" -tokens 160 -top-k 20 -top-p 0.9
go run ./cmd/gofish train -model new-transformer -data my-text.txt -steps 1000 -out transformer.gofish
go run ./cmd/gofish train -model transformer.gofish -data my-text.txt -steps 500 -resume -out transformer.gofish
```

Use `-context`, `-width`, `-heads`, `-layers`, and `-ff` to configure a new
transformer. Width must be divisible by heads. Defaults are context 32, width 32,
heads 4, layers 2, and feed-forward width 128. Training defaults are batch 4 and
learning rate 0.003. `-decay` enables AdamW weight decay. Native transformer
generation counts bytes toward the token budget and constrains decoding to valid
UTF-8, so both the returned text and streamed chunks contain complete characters.

The lower-level `transformer.Model` also exposes differentiable `Forward` and
`Parameters` methods for custom training loops. See [examples/transformer](examples/transformer/main.go).

## Native GGUF Llama models

Run the bundled GGUF demonstration without Ollama, Python or CGO:

```sh
go run ./examples/gguf
go run ./cmd/gofish generate -model tiny-llama -prompt "the little fish" -temperature 0
```

Or load a supported local file:

```go
model, err := gofish.Load("my-model.gguf")
// Optional explicit limits:
model, err = gofish.LoadGGUF("my-model.gguf", gofish.GGUFOptions{
    ContextSize:   512,
    MaxModelBytes: 2 << 30,
    MaxCacheBytes: 256 << 20,
})
```

The loader supports little-endian GGUF v2/v3, architecture `llama`, Llama-style
SentencePiece BPE with byte fallback, RMSNorm, SwiGLU, interleaved RoPE and
grouped-query attention. Encodings are F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1,
Q8_0, Q4_K and Q6_K. Quantized matrices stay packed; CPU matrix-vector operations
decode blocks into scratch storage. There is no GPU acceleration.

Each generation request has a bounded KV cache with absolute rotary positions.
Old entries are evicted when the context fills. Default limits are 2 GiB of
encoded model weights, 256 MiB of cache, and the smaller of declared context and
2048 tokens. `llama.Model.NewSession` exposes incremental token stepping.

This targets the original Llama/TinyLlama layout. GPT-2 tokenizers (including
Llama 3), Qwen/Gemma architectures, MoE, bias tensors, custom rotary-frequency
tensors, noncausal attention, sharded files and RoPE scaling other than none or
linear are rejected. Compatibility is validated with generated GGUF fixtures,
quantization bit-pattern tests, SentencePiece merge tests, and cached logits
compared with a separate full-sequence autograd path. **A downloaded third-party
Llama/TinyLlama checkpoint has not been executed in this environment.**

Inspect a file without loading its weight payloads:

```sh
go run ./cmd/gofish inspect -model my-model.gguf
```

Small GGUF models with at most 2 million parameters support `Train` and `Save`.
Training dequantizes weights into GoFish autograd, starts fresh AdamW state and
exports F32 GGUF; optimizer resume is available in the custom transformer backend,
not GGUF. Large GGUF models expose inference only. `Capabilities()` reports this.
Source license/provenance metadata is retained when supported by the writer;
model licenses remain the user's responsibility.

```sh
go run ./cmd/gofish train -model new-llama -data my-text.txt -steps 500 -out my-model.gguf
go run ./cmd/gofish train -model tiny-llama -data lm/corpora/tiny-stories.txt -steps 100 -out tuned.gguf
```

`gofish.NewLlama` creates a small Llama-shaped byte model. Its tokenizer is the
custom `gofish-byte` type: third-party GGUF runtimes require explicit support for
that tokenizer. The bundled `tiny-llama` uses this type and is an original GoFish
training demonstration, not Meta model weights or a general assistant.

The low-level `gguf` package also provides lazy tensor inspection, bounded reads,
quantized matrix-vector operations, and a deterministic GGUF v3 writer. The
format follows the [upstream GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).

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
Both bundled character models can also be trained further this way. Character
model checkpoints save architecture, vocabulary, and weights; their `Train` calls
always start fresh Adam moments and do not support `Resume` or `WeightDecay`.
All native training steps count updates, not epochs.

CLI equivalents:

```sh
# Train from scratch using the vocabulary in your text.
go run ./cmd/gofish train -model new -data my-text.txt -steps 1000 -out my-model.gofish
go run ./cmd/gofish generate -model my-model.gofish -prompt "hello"

# Continue training a bundled model on compatible text.
go run ./cmd/gofish train -model tiny-stories -data lm/corpora/tiny-stories.txt -steps 100 -out tuned.gofish
```

The character models use a fixed-context embedding/MLP architecture; the byte
model is a decoder transformer. Both are suitable for small CPU experiments.
The custom transformer backend cannot load Hugging Face, PyTorch, or ONNX weights.
The separate native Llama backend reads the supported GGUF subset described below. Limits are 5 million parameters for character models and 2
million for transformers, with an additional training activation budget. The
transformer uses full attention and recomputes its context during generation;
KV caching, GPU training, and fine-tuning Qwen/Llama are not implemented.
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
| `tensor` | Dense float64 tensors, broadcasting, batched matrix multiply, reshape, permute, slice, gather, activations, layer normalization, causal attention, losses, autograd |
| `nn` | Linear, Embedding, Sequential, ReLU, Tanh, Sigmoid, GELU, LayerNorm, MultiHeadAttention, TransformerBlock, weight checkpoints |
| `optim` | SGD, Adam, AdamW, serializable optimizer state, global gradient norm clipping |
| `gguf` | Lazy GGUF reader, writer, dequantization and packed matrix-vector operations |
| `llama` | Native Llama inference, SentencePiece, KV sessions, small-model training and GGUF export |
| `lm` | Native character-model architecture, vocabulary, training and compressed checkpoints |
| `transformer` | Native byte decoder, pretrained weights, training, resumable checkpoints and UTF-8 generation |
| `tokenizer` | Lossless byte tokenization, BOS token and UTF-8 prefix validation |
| `training` | Shared native training options, progress and history |
| `ollama` | Pretrained generation, explicit chat history, streaming and model downloads |
| `generation` | Temperature, top-k, top-p, token filtering and stop-aware streaming |

## Tensor behavior

```go
x := gofish.NewTensor([]float64{2, 3}, 2).WithGrad()
loss := x.Mul(x).Sum()
loss.Backward()
fmt.Println(x.Grad()) // [4 6]
```

Tensors own their storage; `Data`, `Shape`, and `Grad` return copies. Arithmetic
supports NumPy-style broadcasting. `MatMul` broadcasts leading batch dimensions;
`Transpose` swaps 2D matrix axes and `Permute` reorders arbitrary dimensions.
`Slice` selects an interval along an axis. `Softmax` and `LayerNorm` operate on
the last dimension. `Attention` accepts `[batch,time,width]` tensors, splits heads,
and optionally masks future positions before softmax. `CrossEntropy` takes
`[batch, classes]` logits and one integer target per row.

`Backward` accumulates leaf gradients, so call `ZeroGrad` before the next update.
Non-scalar outputs need a same-shaped upstream gradient passed to `Backward`.
Graphs can be reused before parameters change; modifying a value needed by a
pending backward pass produces a panic. `Detach` copies values without their
graph. Shape/index errors panic, while model loading, generation and training
return errors. Higher-order derivatives, sparse tensors, GPU tensors,
convolution, and distributed training are not included.

Individual tensors are not safe for concurrent mutation. High-level native
`Train`, `Generate`, `Loss`, and `Save` calls serialize access to a model; direct
parameter mutation requires your own synchronization.

## Verify and reproduce

```sh
go test -race -cover ./...
go vet ./...
go run ./internal/pretrain
go run ./internal/pretrain-transformer
```

Tests check derivatives with finite differences, shared computation graphs,
broadcasting, embedding gradients, numerical stability, XOR convergence,
language-model convergence, actual bundled checkpoints, save/load parity,
Unicode, cancellation, streaming stops, CLI training, causal masking, transformer
convergence, exact optimizer resume, checkpoint corruption, and Ollama request/response
contracts. Adapter tests use in-process HTTP transports and do not download or
execute a real instruction model.

The three pretraining commands regenerate all four bundled checkpoints from the
included original corpora. There are no external training datasets. GoFish source
code and original bundled materials are licensed under [CC BY-SA 3.0](LICENSE).

The attention calculation follows [scaled dot-product attention](https://arxiv.org/abs/1706.03762);
the native block uses a decoder-only pre-normalization layout. The layer-normalization
implementation uses population variance and learned affine parameters, matching
the [documented LayerNorm definition](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).

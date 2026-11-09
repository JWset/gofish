# GPU training

`NewTransformer` supports NVIDIA CUDA training. Choose the processor in the
same `TrainOptions` you already use:

```go
_, err := model.Train(context.Background(), string(data), gofish.TrainOptions{
    Device: gofish.GPU, // gofish.CPU is the default
    Steps: 5000,
})
if err != nil {
    panic(err)
}
```

Build and run your application with CUDA support:

```sh
go get github.com/JWset/gofish@latest
go mod tidy
go run -tags cuda .
```

Requirements: Go 1.27+, Linux amd64, a CUDA 13 compatible NVIDIA GPU and driver,
CGO enabled, and a C/C++ compiler. The first GPU run downloads and caches the
GoMLX/XLA CUDA runtime and its NVIDIA libraries, which can take several minutes
and significant disk space. Python is not required: Go downloads and extracts
runtime packages. Later runs reuse the cache. See the upstream
[GoMLX backend documentation](https://gomlx.github.io/docs/backends/).

The GPU runs the forward pass, backpropagation, gradient clipping and AdamW
updates in float32. Weights and optimizer moments remain on the GPU between
updates; sampled token batches are uploaded, and loss/progress values are read
back. On return, weights and moments are copied to GoFish's native model so
`Save`, `Generate` and subsequent CPU training work normally. `Resume: true`
works across saved checkpoints and device switches. Expect small numerical
differences between float32 CUDA and float64 CPU training; switching devices
is not an exact continuation in floating-point arithmetic.

`Device: gofish.CPU` (or omitting `Device`) uses CPU training, including in a
CUDA-enabled build. A GPU request without the CUDA build or a usable CUDA device
returns an error. It never silently falls back to CPU.

Currently GPU training applies to `NewTransformer` and loaded byte-transformer
checkpoints. Character models and GGUF/Llama training return an unsupported
GPU error. Generation remains on CPU for the native models. The existing
2-million-parameter transformer and activation limits still apply. Tiny models
may be slower on GPU because launch and compilation overhead dominate; this change
does not enable full training of 4-billion-parameter models.

From the GoFish repository, the CLI also accepts the device selection:

```sh
go run -tags cuda ./cmd/gofish train -model new-transformer \
  -device gpu -data training.txt -steps 5000 -out aot.gofish
```

For a hardware integration check on an NVIDIA machine:

```sh
GOFISH_CUDA_TEST=1 go test -tags cuda ./transformer -run '^TestCUDA' -timeout 60m -v
```

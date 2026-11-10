# Changes

## Local open-weight pretrained models

- Add a managed, authenticated loopback llama.cpp backend for Qwen3, Gemma and SmolLM GGUF families.
- Preserve native models and route supported pretrained GGUF files through the existing Load/LoadGGUF, Generate and Chat APIs.
- Add context, runtime path, GPU layer, template and startup controls; add cancellable loading and explicit Close.
- Add runtime-backed tokenization/template methods, structured function calls, JSON-schema output and streaming deltas.
- Add native and constrained-JSON tool modes, per-architecture contract tests, pinned real-checkpoint integration tests and provenance hashes.

## CUDA training

- Add `TrainOptions.Device`: `gofish.CPU` (default) or `gofish.GPU`.
- CUDA builds train byte transformers with float32 forward/backprop, clipping and AdamW on NVIDIA GPUs through GoMLX/XLA.
- Preserve native checkpoint compatibility and optimizer resume across devices.
- Add CLI `train -device cpu|gpu` and [GPU setup instructions](GPU.md).
- Require Go 1.27 for the pinned accelerator dependencies.

## Native GGUF expansion

- Add bounded GGUF v2/v3 parsing, lazy tensor reads, a v3 writer and ten weight encodings.
- Add native Llama inference with RMSNorm, RoPE, GQA, SwiGLU and a bounded KV cache.
- Add SentencePiece BPE with byte fallback and explicit unsupported-variant errors.
- Fine-tune small GGUF models and export F32 weights; large models are inference-only.
- Bundle an original trained GGUF fixture and add file inspection to the CLI.
- Check cache/full-sequence parity, quantization layouts, export/reload, training and parser fuzz seeds.

## Native transformer expansion

- Add a trainable byte-level decoder with causal multi-head attention, learned
  positions, pre-layer normalization, GELU, and residual paths.
- Bundle `tiny-transformer`, a 34,881-parameter checkpoint trained for 6,000 updates.
- Use the same high-level load, train, save, loss, and generation API for both
  native architectures, with automatic checkpoint-format detection.
- Add AdamW and serializable optimizer moments. Transformer checkpoints resume
  exactly with the same corpus and training options.
- Add batched/broadcasted MatMul, arbitrary Permute, Slice, GELU, LayerNorm,
  and fused differentiable causal or noncausal attention to the tensor core.
- Add composable LayerNorm, MultiHeadAttention and TransformerBlock modules.
- Add byte tokenization and UTF-8 constrained generation.
- Add top-k and nucleus sampling to native generation and the Ollama adapter.
- Extend the CLI with transformer creation, architecture flags and optimizer resume.
- Add numerical gradient checks, future-token leakage checks, training convergence,
  exact checkpoint resume, corrupted-checkpoint rejection, Unicode generation,
  concurrent access and CLI transformer tests.

## Initial foundation

- Float64 CPU tensors, broadcasting and reverse-mode autograd.
- Linear and embedding layers, sequential networks, losses, SGD and Adam.
- Two trainable bundled character models, native checkpointing and examples.
- Ollama generation, chat, model downloads and stop-aware streaming.
- A Go module, command-line interface, CI workflow and publishing instructions.

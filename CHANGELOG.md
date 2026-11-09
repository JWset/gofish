# Changes

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

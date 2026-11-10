# Compatibility run: 2026-11-10

All nine targets in `compatibility.json` passed with no skipped targets.
Model SHA-256 values were checked against the pinned manifest before inference.

- Runtime: llama.cpp b12242, commit 3b4d6ab3f (0.9.0-dev).
- Platform: Linux amd64, CPU inference, four threads, context 512.
- Command: `GOFISH_REQUIRE_ALL_GGUF=1 GOFISH_VERIFY_GGUF_SHA=1 go test -race ./pretrained -run TestGGUFCompatibility -v -timeout 20m`, with `GOFISH_LLAMA_SERVER` and `GOFISH_GGUF_DIR` pointing to local files.
- Results: Qwen3 0.6B, 1.7B, 4B; Gemma 1.1 2B, Gemma 2 2B, Gemma 3 270M; SmolLM 135M, SmolLM2 135M, SmolLM3 3B: PASS.
- Quantizations exercised: Q8_0 and mixed Q4_K/Q6_K (Q4_K_M checkpoints), with F32 tensors.
- Each target: Unicode tokenizer behavior, embedded chat template, streamed raw completion, streamed chat, streamed schema output, typed function calls/deltas, tool-result follow-up, and idempotent close.

Qwen3 uses native tool templates. Gemma and SmolLM use constrained JSON tool
calling. Gemma 1 uses its original conversation template with raw constrained
completion for JSON output, avoiding a legacy chat-parser grammar issue.
SentencePiece dummy-prefix normalization is tested explicitly for Gemma 1.

This is a file/runtime compatibility check, not a quality benchmark, training
claim, GPU benchmark, or blanket guarantee for other GGUF conversions or runtime
versions. Model weights and runtime binaries are not distributed in this repo.

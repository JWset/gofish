# Bundled native models

GoFish includes two trained checkpoints, not randomly initialized placeholders.
They ship as embedded gzip-compressed JSON under `lm/assets` and run offline.

| Model | Corpus | Intended use |
| --- | --- | --- |
| `tiny-stories` (alias `tiny`) | `lm/corpora/tiny-stories.txt` | Continue a short story about a fish and a bird |
| `tiny-dialogue` | `lm/corpora/tiny-dialogue.txt` | Reproduce patterns from a small set of dialogue examples |

These are demonstrations of loading, differentiable training, text sampling,
and fine-tuning. **They do not understand general questions or provide reliable
answers.** The dialogue model is not a general assistant. For general-purpose
text generation, use a separately downloaded pretrained instruction model
through the Ollama backend.

## Architecture and training

- Rune-level vocabulary derived from each corpus; NUL is reserved for padding.
- 12-character context window, 8-dimensional learned embeddings.
- Flattened embeddings, a 64-unit tanh layer, and a vocabulary-sized output layer.
- Next-character cross entropy, Adam with learning rate 0.01.
- Global gradient norm clipped to 1; batches of 32 random corpus positions.
- 1,800 updates per model; initialization and training RNG seeds both 42.
- CPU float64 operations using GoFish's own autograd implementation.

Recorded mean cross entropy over the training corpus after the initial build:

| Model | Before training | After training |
| --- | ---: | ---: |
| `tiny-stories` | 3.3701 | 0.0222 |
| `tiny-dialogue` | 3.4427 | 0.0084 |

Loss is in natural-log units. These are **training-set measurements**, with no
held-out evaluation. The models largely memorize their tiny corpora. They may
loop, mix unrelated fragments, and generate nonsense on unfamiliar contexts.
The fixed vocabulary rejects new characters rather than silently losing them.

The corpora were authored for this project and are included under its MIT
license. No external model weights or scraped datasets are bundled. Ollama
models are separate downloads and retain their upstream licenses.

## Reproduction

From the repository root:

```sh
go run ./internal/pretrain
go test ./lm
```

The checkpoint includes its configuration, vocabulary and trained weights.
Optimizer state is not saved. Reloading and calling `Train` fine-tunes the
weights with a fresh Adam optimizer.

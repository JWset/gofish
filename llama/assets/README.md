`tiny-llama.gguf` is an original small Llama-shaped model trained on the included
fish-story corpus using GoFish. It has standard GGUF architecture `llama`, F32
weights, and the custom `gofish-byte` tokenizer. It is a compatibility fixture
and text-generation demo, not Meta Llama weights or a general-purpose assistant.
Rebuild it with `go run ./internal/pretrain-llama`.

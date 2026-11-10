# Local pretrained GGUF models

GoFish keeps its native transformer and CPU/CUDA training API. Pretrained
inference uses an optional, managed **local llama.cpp `llama-server` process**.
This provides the actual architecture implementations, tokenizers, quantized
kernels and Jinja conversation templates. It is not a pure-Go reimplementation
of these pretrained architectures, and it does not need Ollama or a cloud API.

## Setup

Install/extract `llama-server` and its accompanying libraries for your platform
from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases).
The compatibility matrix below was tested with **b12242 / 3b4d6ab3f**, CPU
Linux amd64. Preserve the runtime's accompanying license files. GPU offload
requires an appropriate CUDA, Metal, Vulkan or other GPU runtime build.

Put `llama-server` on PATH, set `GOFISH_LLAMA_SERVER` to its executable path, or
pass `GGUFOptions.ServerPath`. Download a compatible GGUF separately; GoFish
never silently downloads weights or accepts a model license for you.
Upstream weights retain their own licenses and are not bundled with GoFish.

```go
ctx := context.Background()
model, err := gofish.LoadPretrained(ctx, "Qwen3-4B-Q4_K_M.gguf", gofish.GGUFOptions{
    ContextSize: 4096,
    ServerPath: "/path/to/llama-server", // optional if on PATH or set through the environment
    // GPULayers: 99, // optional: requires a GPU-enabled runtime
})
if err != nil { panic(err) }
defer model.Close()

text, err := model.Chat(ctx, []gofish.Message{
    {Role: "system", Content: "Be concise and distinguish facts from uncertainty."},
    {Role: "user", Content: "Explain how a transformer works."},
}, gofish.WithMaxTokens(200), gofish.WithStream(func(chunk string) error {
    _, err := fmt.Print(chunk)
    return err
}))
if err != nil { panic(err) }
_ = text // complete generated text, also returned when streaming
```

`Load(path)` detects GGUF. `LoadGGUF(path, options)` automatically selects the
native backend for native fixtures and the managed backend for supported
pretrained architectures. `LoadGGUFContext(ctx, path, options)` adds cancellable
startup. `LoadPretrained` explicitly selects the managed backend. Set
`Runtime: "native"` in `LoadGGUF` to explicitly request the older native Llama
subset; its tokenizer, architecture and training restrictions still apply.

The server binds only to 127.0.0.1, uses a random API key, disables outbound
model fetching, and has one inference slot. GoFish bypasses HTTP proxies and
rejects redirects. Prompts are sent to this local process. Call `Close()` when
finished; it cancels active requests and terminates the managed server. It is
safe to call twice. Native models and Ollama clients have a no-op `Close()`.

## Tokenizers, context and templates

`Tokenize(ctx, text)` and `Detokenize(ctx, ids)` use the loaded model's tokenizer,
including its vocabulary, BPE merges/SentencePiece rules and special-token IDs.
Plain `Tokenize` neither adds BOS nor interprets special-token spellings.
SentencePiece normalization can introduce a leading dummy space (notably Gemma
1); raw detokenization preserves that behavior. `PretrainedInfo()` reports it
in `TokenizerAddsSpacePrefix`. GoFish never fixes this by trimming user text.

`ApplyChatTemplate(ctx, messages)` shows the actual runtime-formatted prompt.
`Chat` uses the GGUF's embedded `tokenizer.chat_template`. A model without a
chat template can still use raw `Generate`; chat returns an actionable error.
Supply a Jinja **file path** in `GGUFOptions.ChatTemplate` to override a missing
or unsuitable template. Role restrictions of the template still apply.

Context defaults to min(declared context, 2048). Explicit `ContextSize` must fit
the GGUF's declared context. Context extension/RoPE extrapolation is not enabled
implicitly. `MaxModelBytes` defaults to a 16 GiB file-size cap and
`MaxCacheBytes` to a 1 GiB estimated FP16 KV-cache budget for this backend.
The cache estimate is conservative for sliding-window models and excludes
weights, work buffers, activations and GPU driver memory; it is not a hard
process RSS cap. Increase the budget explicitly or reduce context when needed.
The existing native backend keeps its previous default budgets.

Qwen3 chat requests set the template's `enable_thinking` flag to false.
`Generate` produces a raw continuation without applying chat formatting.
A context overflow, invalid weight layout, unsupported quantization, tokenizer
or template produces a load/request error with the runtime diagnostic.

## Structured function calling

```go
tools := []gofish.Tool{{
    Type: "function",
    Function: gofish.Function{
        Name: "lookup_lore",
        Description: "Look up a passage in the local lore collection",
        Parameters: json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}},"required":["query"],"additionalProperties":false}`),
        Strict: true,
    },
}}
messages := []gofish.Message{{Role: "user", Content: "Look up the history of the walls."}}
response, err := model.ChatStructured(ctx, messages, gofish.ChatOptions{
    Tools: tools,
    ToolChoice: "required", // also "auto" (default) or "none"
    OnDelta: func(delta gofish.ChatDelta) error {
        // delta.Content, delta.ReasoningContent, and delta.ToolCalls
        return nil
    },
}, gofish.WithMaxTokens(256))
if err != nil { panic(err) }
messages = append(messages, response.Message)
for _, call := range response.Message.ToolCalls {
    // call.Function.Name and call.Function.Arguments (a JSON object string)
    // Validate arguments/domain rules, execute your own function, then append:
    messages = append(messages, gofish.Message{
        Role: "tool", ToolCallID: call.ID, Content: `{"passage":"your lookup result"}`,
    })
}
answer, err := model.Chat(ctx, messages, gofish.WithMaxTokens(200))
_ = answer
_ = err
```

GoFish returns typed calls and never executes a function automatically. It
rejects undeclared function names, malformed JSON argument objects, missing
call IDs and missing required calls. The application remains responsible for
full argument validation, permissions and side effects.

`ChatOptions.ToolMode` can be `auto`, `native`, or `json`. Auto uses Qwen3's
native tool template/parser; Gemma and SmolLM use a JSON-schema-constrained
function envelope because their stock templates/checkpoints do not reliably
follow native tool requests. JSON mode constrains the function name and
arguments schema and translates call/result history to ordinary conversation
turns. It produces one function call per response. Allow enough output tokens;
truncated JSON or incomplete calls return an error. Model quality still affects
argument values; constrained JSON does not make a model knowledgeable.

Native mode streams partial tool names/argument fragments through `OnDelta`.
JSON mode buffers and validates the envelope, then emits a complete tool-call
delta atomically. Ordinary text generation/chat streams live for all families.
`WithStream` receives text only. Callback errors and canceled contexts stop
requests; a truncated SSE stream returns an error rather than silent success.

`ChatOptions.ResponseSchema` provides JSON-schema-constrained ordinary output.
It cannot be combined with `Tools`; tool mode supplies its own schema.
For Gemma 1, schema output applies the original chat template before raw
constrained completion, avoiding the legacy chat-parser grammar failure.
Gemma/SmolLM's typed tool history is translated for their stock templates even
when the follow-up `Chat` request no longer supplies function definitions.

## Compatibility targets and tests

[All nine targets passed on 2026-11-10](pretrained/testdata/COMPATIBILITY.md).
The pinned source revision and SHA-256 for each file are recorded in
[the machine-readable manifest](pretrained/testdata/compatibility.json).
`gofish.PretrainedModels()` lists these reference checkpoints programmatically.

| Target | GGUF architecture | Reference checkpoint | Tool mode |
|---|---|---|---|
| Qwen3 0.6B | qwen3 | Qwen/Qwen3-0.6B-GGUF · Q8_0 | native |
| Qwen3 1.7B | qwen3 | Qwen/Qwen3-1.7B-GGUF · Q8_0 | native |
| Qwen3 4B | qwen3 | Qwen/Qwen3-4B-GGUF · Q4_K_M | native |
| Gemma 1 | gemma | bartowski/gemma-1.1-2b-it-GGUF · Q4_K_M | JSON |
| Gemma 2 | gemma2 | bartowski/gemma-2-2b-it-GGUF · Q4_K_M | JSON |
| Gemma 3 (text) | gemma3 | unsloth/gemma-3-270m-it-GGUF · Q8_0 | JSON |
| SmolLM 1 | llama | HuggingFaceTB/smollm-135M-instruct-v0.2-Q8_0-GGUF | JSON |
| SmolLM 2 | llama | bartowski/SmolLM2-135M-Instruct-GGUF · Q8_0 | JSON |
| SmolLM 3 | smollm3 | unsloth/SmolLM3-3B-GGUF · Q4_K_M | JSON |

Each integration target checks actual loading, Unicode tokenization and
model-specific normalization, conversation formatting, quantized raw generation,
streamed chat, constrained JSON, structured function calls/deltas, a subsequent
tool-result conversation, and close behavior. The ordinary suite separately
checks architecture/tokenizer contracts, budgets, unsupported architectures,
SSE fragmentation/truncation, callback cancellation and process startup cleanup.
These checks establish file/runtime compatibility; they are not model quality
benchmarks and do not guarantee that every third-party GGUF conversion works.

To reproduce, place those exact manifest filenames in one directory and run:

```sh
GOFISH_LLAMA_SERVER=/path/to/llama-server \
GOFISH_GGUF_DIR=/path/to/gguf-files \
GOFISH_REQUIRE_ALL_GGUF=1 GOFISH_VERIFY_GGUF_SHA=1 \
go test ./pretrained -run TestGGUFCompatibility -v -timeout 20m
```

For a subset/custom conversion, `GOFISH_GGUF_MODELS` is a JSON map from target
name (for example `qwen3-0.6b`) to local filename. Missing targets skip unless
`GOFISH_REQUIRE_ALL_GGUF=1` is set. Real checkpoints are never downloaded by the
test suite and are excluded from the GoFish repository.

Only the architecture allowlist above plus the existing native Llama subset is
supported. Qwen3 MoE, Qwen3.5, Gemma 3n/4, vision/projector input, embeddings,
sharded GGUFs and other architectures return explicit unsupported errors.
Gemma 3 support is **text only**. Pretrained managed models are inference-only;
they do not implement GoFish training, optimizer checkpoints, or weight export.

## CLI

```sh
go run ./cmd/gofish chat -model /path/to/Qwen3-0.6B-Q8_0.gguf \
  -llama-server /path/to/llama-server -context 2048

go run ./cmd/gofish generate -model /path/to/model.gguf -prompt 'Once upon a time' \
  -llama-server /path/to/llama-server -gpu-layers 99
```

The `-tags cuda` build flag belongs to native GoFish GPU **training**. Managed
pretrained inference uses the capabilities of the chosen `llama-server` binary
and requires no Go build tag.

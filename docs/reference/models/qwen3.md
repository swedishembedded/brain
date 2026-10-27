# Qwen3

Qwen3-0.6B, a dense instruct/chat decoder (grouped-query attention, QK-norm,
RoPE, SwiGLU MLPs) running on brain's own compute engine. This is brain's
flagship served LLM: it's the model behind brain's OpenAI/Anthropic/
OpenRouter-compatible HTTP endpoints and its D-Bus surface, with concurrent
request batching and a paged KV cache. Reach for it for chat and tool-calling
inference, training from scratch, or LoRA finetuning a named adapter on your
own data.

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| Training from scratch | [x] |
| LoRA fine-tune         | [x] |
| INT8                   | [x] |
| CLI (`brain <arch> <action>`)       | [x] |
| HTTP API               | [x] |
| D-Bus                  | [x] |
| Batched serving        | [x] |

## Getting the weights

`Qwen/Qwen3-0.6B` is auto-fetched from Hugging Face on first use (opt-in: `--autofetch`). You can
also import a checkpoint you already have locally:

```bash
brain qwen3 import --hf /path/to/Qwen3-0.6B --out qwen.safetensors
```

A **GGUF** works too -- llama.cpp's `qwen3` architecture, e.g.
`Qwen/Qwen3-8B-GGUF` -- and needs **no conversion step**: `qwen3` is one of
the architectures brain's model-dir scan and every `qwen3` CLI verb load
**directly**. `checkpoint::weightio::WeightReader::open` sniffs the file's
`GGUF` magic (falling back to the `.gguf` extension) and dispatches straight
to the GGUF reader, with no intermediate `.safetensors` in between. Point
`--weights` straight at the file:

```bash
brain qwen3 infer --weights /path/to/Qwen3-8B-Q8_0.gguf \
    --tokenizer tokenizer.json --prompt "Hello"
```

and the same goes for `brain serve`'s resident (below) -- `--checkpoint`
also accepts a `.gguf` path directly.

The generic converter is still there if you want it -- `brain import
/path/to/Qwen3-8B-Q8_0.gguf --out qwen3-8b.safetensors`, which picks the
right converter from the file's own `general.architecture` -- but for `qwen3`
it is now **optional**, not a prerequisite: reach for it when you actually
need a brain-native `.safetensors` checkpoint, e.g. as the base for
`finetune`/`train` (a GGUF has no training path) or to pin one canonical
on-disk format for the model store. Both routes produce the same brain
parameters when you do convert. The tensor-name map is transcribed from
llama.cpp's own `gguf-py/gguf/tensor_mapping.py` and `constants.py`, and a
test asserts the two routes agree *bit for bit* on the same logical
checkpoint -- not to a tolerance, because the mistake worth catching (a
swapped `k`/`v` projection) is shape-compatible on every GQA layer and only an
exact comparison sees it.

Converting still pays the same disk trade, if you choose to: brain's own
checkpoint format is fp32, so converting a quantized GGUF *expands* it -- a
Q8_0 file is a much smaller download than the bf16 safetensors, and the fp32
conversion is larger than either. Running the GGUF directly, as above, never
pays that expansion at all -- the download saving is the whole story, the
same as FLUX.2's text encoder, below.

To make a checkpoint a `brain serve` resident, pass `--checkpoint PATH`
(`--tokenizer PATH` names a tokenizer that is not beside the checkpoint) -- a
`.gguf` file works here too. A checkpoint in the model store needs neither: the
store scan serves it under its store id.

### As FLUX.2's text encoder

FLUX.2 Klein conditions on a Qwen3, so `BRAIN_FLUX2_TE` accepts either an HF
text-encoder **directory** or a Qwen3 **`.gguf` file**. Which one a path is, is
sniffed from the path and its contents -- there is no flag to set, and
directory behaviour is unchanged.

## Running it

```bash
# Inference
brain qwen3 infer --weights qwen.safetensors --prompt "Hello"

# Training from scratch
brain qwen3 train --data data/shakespeare_char --out qwen-train.safetensors --steps 1000

# LoRA finetune -- trains a NAMED adapter
brain qwen3 finetune --lora 8 --weights Qwen/Qwen3-0.6B \
    --adapter my-org/my-finetune --dataset /path/to/dataset/dir --steps 500

# Prove it learned: held-out loss/accuracy, base vs. the adapter
brain qwen3 eval --weights Qwen/Qwen3-0.6B \
    --adapter my-org/my-finetune --jsonl /path/to/validation.jsonl

# Full-parameter finetune from a plain checkpoint file (no adapter)
brain qwen3 finetune data/shakespeare_char --weights qwen.safetensors \
    --out qwen-ft.safetensors --steps 500

# Paged-KV continuous-batching decode over a batch of prompts (a CLI demo of
# the serving engine, not an HTTP server -- see `brain serve --openai PORT`
# for that)
brain qwen3 serve --weights qwen.safetensors --tokenizer tokenizer.json \
    --prompt "Hello" --prompt "Hi there"

# Device selection
brain qwen3 infer --weights … --device cpu     # JIT-compiled CPU path
brain qwen3 infer --weights … --device gpu     # portable GPU backend (default)
brain qwen3 infer --weights … --device vulkan  # native Vulkan
```

Other verbs: `brain qwen3 export` (to ONNX, or `--format hf|gguf` for a
`transformers` directory or a llama.cpp GGUF, see below), `brain qwen3 precompile` (precompile
kernels for a target device), `brain qwen3 toolcall` (tool-call evaluation).

### Exporting to Hugging Face

```bash
brain qwen3 export --weights model.safetensors --format hf --out my-model \
    --tokenizer-dir path/to/base-checkpoint [--dtype bf16|f16|f32]
```

writes a directory `transformers` loads with `from_pretrained`: sharded
safetensors (5 GB shards and an index, or one `model.safetensors`) under the
HF tensor names, and a `config.json` naming the class the checkpoint is:
`Qwen3ForCausalLM` with QK-norm, `Qwen2ForCausalLM` with q/k/v bias,
`LlamaForCausalLM` with neither, RoPE scaling included. The tokenizer,
chat-template and generation files are copied from `--tokenizer-dir`, which
defaults to `--weights` when that is a checkpoint directory. Any checkpoint
the decoder reads can be the input: a brain file, a Hugging Face directory or
a GGUF. Tensors stream one at a time. The default dtype is bf16; `--dtype
f32` exports losslessly. `--adapter ADAPTER.safetensors` folds a LoRA adapter
into the weights it targets as they are written.

`--format gguf --out model.gguf [--dtype f16|f32]` writes a llama.cpp GGUF
under the architecture the checkpoint is (`qwen3`, `qwen2`, `llama`), with its
tokenizer embedded, which llama.cpp needs. A Llama's q/k rows are stored in
llama.cpp's interleaved RoPE order. A llama3 RoPE scaling becomes the
`rope_freqs.weight` divisors. The tokenizer's pre-tokenizer is identified from
its `tokenizer.json`, and one this exporter does not know is refused rather
than guessed. Norms and biases stay f32, as llama.cpp writes them; the
default dtype for the matrices is f16. Exporting deepseek-coder-1.3b-instruct
gives tensors byte-identical to llama.cpp's own converter, the same metadata,
and the same tokenization under llama.cpp.

`--format peft --adapter ADAPTER.safetensors --out DIR [--base-model HF_ID]`
writes a LoRA adapter as a PEFT adapter directory (`adapter_model.safetensors`
and `adapter_config.json`, with `lora_alpha` and the target modules) that
`peft.PeftModel.from_pretrained` applies to the HF export of its base.
`--base-model` defaults to the base on the adapter's card.

### Full fine-tuning

`brain qwen3 finetune <data_dir> --weights BASE --out F` trains every weight.
`BASE` is a brain checkpoint, a GGUF, a `transformers` directory or a model
store `vendor/repo`, read as it is on disk and streamed into the training
build: nothing is converted or written first. The AdamW moments live in host
RAM, so the card holds the weights and their gradients (fp32): a 1.3B-1.5B
decoder trains on one 24 GB card. `F` is a brain checkpoint (rerun with the
same `--out` to continue it; `brain qwen3 export` writes it back as a
`transformers` directory). `<data_dir>` is a token dataset
(`train.u32.bin`, `val.u32.bin`, `meta.json`). `--activations keep|recompute`
sets what the backward keeps of the forward; the decoder always recomputes each
layer, so `keep` is refused by name.

### LoRA adapters

A named LoRA adapter is stored beside its base checkpoint in the model store,
at `<models-dir>/Qwen/Qwen3-0.6B/adapters/<owner>/<name>/<tag>/`, and is
addressed as `Qwen/Qwen3-0.6B:<owner>:<name>:<tag>`. The tag defaults to
`latest` and is **overwritten on every rerun** of the same adapter name - a
finetune run always fully retrains and replaces that tag, it never
incrementally continues a previous run.

The base may be a brain checkpoint or a `transformers` directory
(`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, ...), and `brain serve` lists the
adapter as its own model id beside the base, folding it into the base's
weights when it is loaded.

## Options

Which checkpoint to serve is the model store, or `brain serve --checkpoint`
(with `--tokenizer`); FLUX.2's text encoder still takes `BRAIN_FLUX2_TE`,
above. Everything about HOW to serve it is a `brain serve` flag:

- `--lora N` - LoRA rank for `finetune`.
- `--lora-targets LIST` - the projections a `finetune --lora` adapter
  covers, comma-separated from `wq,wk,wv,wo,gate,up,down` (the default is
  all seven). An unknown or repeated name is refused.
- `--grad-accum N` - micro-batches averaged into each step of a `finetune
  --lora` run. On a model laid out as a pipeline across cards the stages work
  on neighbouring micro-batches at once instead of one after another, and the
  run ends where the sequential schedule would.
- `--base-format f32|bf16|int8|nf4|fp8` (earlier spelling `--base-dtype`) - the storage format of the frozen base during
  a `finetune --lora` run (default `f32`). `bf16` halves the base's bytes - a
  7B decoder trains on one 24 GB card - and holds the frozen embedding and
  LM head in bf16 too, while the adapters, activations and optimiser stay
  fp32. `int8` holds each linear as a byte per weight and a scale per 32
  (about 7.6 GB for a 7B decoder, against 15 GB), with the same bf16 tables:
  the matrix kernels decode the weight as they load it and multiply it with
  fp32 activations, in the forward and in the input gradient, so the only
  approximation is the weight's own rounding. `nf4` (4-bit NormalFloat, a scale per 32)
  and `fp8` (block-scaled E4M3) run through the shared quantized-execution
  contract: a bounded tile decode feeds the dense product, forward and input
  gradient. `--weights` may name a `transformers` checkpoint
  directory (a model-store `vendor/repo`): it is read as downloaded, one
  tensor at a time, and nothing is written beside it.
  A model that does not fit one card is laid out across the cards by what
  each has free, with no flag: the whole model on one card when one has room,
  otherwise a pipeline of the fewest stages that fit (printed as `pipeline of
  N stages`), otherwise a refusal naming the bytes. An explicit `--device`
  pins one card. A 7B decoder with a bf16 base trains on one 24 GB card to
  about 3k tokens, and at 7.6k tokens across two.
- `--keep-reasoning` - train each answer's `<think>...</think>` reasoning.
  A reasoning model's chat template drops it from the assistant turns it
  renders as history (DeepSeek-R1's drops it from every turn), which is
  right when serving and wrong when the reasoning is what the data teaches.
  The `lora_train` action takes it as `keep_reasoning`.
- `--weight-decay W`, `--grad-clip C` (0 disables), `--warmup N`,
  `--min-lr X`, `--beta1 B`, `--beta2 B`, `--adam-eps E` - the LoRA
  finetune schedule and AdamW settings. The defaults are 0.1, 1.0, 5% of the
  steps, `lr/10`, and torch's AdamW (0.9, 0.999, 1e-8). The `lora_train`
  action takes the same settings as `targets`, `weight_decay`, `grad_clip`,
  `warmup`, `min_lr`, `beta1`, `beta2` and `eps`.
- `--activations keep|recompute` - what the backward keeps of the forward; the
  `lora_train` action takes it as `activations`. The decoder always recomputes
  each layer, so `keep` is refused by name before the first step.
- `--device cpu|gpu|vulkan` - backend selection.
- `--ctx N` - built context length. Unset by default: the resident
  auto-sizes it to the target device's real usable VRAM instead of a fixed
  number, so raising the ceiling normally means freeing VRAM (or adding a
  card), not passing this flag. Set it to pin an exact value instead of the
  auto-picked one. The auto-sized value never exceeds the checkpoint's
  declared `max_position_embeddings`; an explicit value past it derives a
  YaRN (arXiv 2309.00071) RoPE scaling (Qwen's published recipe: `factor =
  ctx / pretrained window`, the pretrained window being the largest power of
  two not above the declared one, so `--ctx 131072` on Qwen3 is factor
  4.0 over 32768) - a checkpoint that already declares one is never
  second-guessed. YaRN rescales every request, short ones included.
- `--max-batch N` - concurrent serving batch slots (default 16).
- the KV pool size - KV pool size in tokens, shared by every sequence
  and by the prompt-prefix cache (default: room for `--max-batch` half-context histories when
  that fits the card beside the weights, else the largest half of it that does,
  else two contexts). An agent that
  resends its whole history every turn is served from the prefix cache only
  while its history stays cached between turns, so size this at clients x
  history tokens (int8 KV costs about 74 KB per token on the 8B, so 250000
  tokens is about 18 GB). A pool is never smaller than one context.
- `--kv-cache f32` - opt out of the default int8 KV cache.
- `--kv-cache fp8` - store the packed KV pools as FP8 E4M3 instead of int8
  (the same bytes per token; the tensor-core prefill decodes the E4M3 bytes
  while staging; off by default).
- `--kv-calibration` - opt in to per-head KV clip ranges produced by `brain
  qwen3 calib` (off by default).
- `--kv-offload-gb N` - host RAM the serving engine may use to park
  preempted sessions' KV cache (default `0`, off). See below.
- `--weight-format int8` / `--weight-format f32` - quantize the 7
  per-layer linears to int8, or keep them fp32. Unset, a checkpoint of 6B
  parameters or more is quantized (its fp32 weights do not fit one 24 GB
  card) and a smaller one is not - at the real Qwen3-8B config, int8
  shrinks the weight term from ~20.8 GiB to ~11.9 GiB. A device with no packed-int8 dot path (the CPU
  backend, or an unusual GPU) degrades to fp32 with a printed warning
  rather than failing.
- `--weight-format bf16` - store the linears and the head as bf16 (exact
  for a checkpoint that ships bf16, half the fp32 bytes in memory), decoded
  inline.
- `--gemm fp32` - pin exact fp32 arithmetic for the linears. By
  default the linears of fp32 weights with at least 8 rows (prefill and
  batched decode) run on TF32 tensor cores where the card has them
  (operands rounded to a 10-bit mantissa, fp32 accumulation, no extra
  memory).
- the prefill row cap - cap the chunked-prefill row count below its
  512 default (clamped to `1..=512`; can only shrink it). The scores/probs
  scratch buffer this bounds is the single largest per-token-scaling cost
  in the whole engine - larger than the KV pool itself - and shrinks
  LINEARLY with this value: 512→128 saves ~2.25 GiB at ctx=24576 on the
  real Qwen3-8B config, with no effect on decode throughput or the KV
  pool's own size (only the prefill chunk shape changes: more, smaller
  passes over the same prompt).

## Serving more sessions than the KV pool holds

The KV pool is sized for a fleet of half-length histories (see the KV pool size), while the batch
has `--max-batch` slots (default 16), so a busy server can admit more sessions
than the pool can keep cached at once. Admission only checks that a prompt
fits, so the sequences it lets in can still outgrow the pool together while
they decode. Whatever the offload setting, that never fails a request: the
scheduler preempts the newest sequences until the rest can append their next
token. Without host offload (the default) a preempted sequence gives its blocks
back, goes to the head of the waiting queue with the tokens it has produced and
its sampler state, and prefills its context again once the pool has room; the
batch stays as full as the pool allows and every request still produces the
tokens it would have produced alone. That costs a re-prefill per preemption,
which host offload below avoids. A request whose prompt plus `max_tokens`
cannot fit the whole pool even alone is refused up front with a clear error.

Give the engine host RAM and it swaps instead (recomputing only when the host
budget is spent). A session the scheduler is
not advancing this round has its whole KV copied to host memory and its GPU
blocks handed to the sessions that are decoding; when its turn comes round
again the bytes are copied back and it continues **exactly** where it left
off - the restored cache is byte-identical, so the tokens it goes on to
produce are identical to the ones it would have produced had it never been
preempted. That is a tested property, not an aspiration.

```bash
brain serve --openai 8080 --kv-offload-gb 16
```

What this buys is **concurrency**, not a longer single context. A session
that is actively decoding still needs its whole KV in VRAM, because causal
attention reads all of it on every token - there is no cache tier that helps
there, and a design that streamed blocks in per token would be far slower
than not extending the context at all (the host bus is one to two orders of
magnitude slower than the card's own memory; `crates/gpu-core/tests/pcie_handoff.rs`
measures both on your hardware). What offload removes is the requirement that
every ADMITTED session be resident simultaneously.

It is off by default because it spends host memory, and a box with none to
spare should not have it taken silently. Budget roughly the same bytes per
cached token that the GPU pool spends - a swap is a verbatim copy.
Offload also keeps a small device staging buffer (at most 64 MiB) that a swap
moves blocks through; it is allocated when the engine loads and counted in the
memory budget (and in `--ctx` auto-sizing), so the first preemption never
asks the card for memory it does not have.

## Hardware and limits

The serving engine doesn't yet reuse a shared prompt prefix across
separate requests (each request's prefill is independent). Mixture-of-experts
style configs are defined in the parameter layout but the serving engine
currently serves dense configs only.

## Serving an embedding checkpoint

A downloaded Qwen3-Embedding checkpoint (a repository whose name contains
`embedding`; its config is the chat model's) is served for embeddings, not chat.
`brain serve --openai` lists it with the `embeddings` capability and answers
`POST /v1/embeddings` for it by the model's own pooling: the hidden state of the
last token, with an end-of-text token appended, L2-normalised (dimension 1024
for Qwen3-Embedding-0.6B). Inputs are embedded as plain passages; the
asymmetric query instruction of the model card is the `instruction` param of the
native `embed` action and is not reachable through the OpenAI route. The
checkpoint loads on the first request and is not counted against the residency
budget until then. A chat request for it is refused: it has no `generate`
action.

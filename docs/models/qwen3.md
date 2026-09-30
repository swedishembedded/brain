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

and the same goes for `brain serve`'s resident (below) -- `BRAIN_QWEN_WEIGHTS`
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

To make a checkpoint a `brain serve` resident, point `BRAIN_QWEN_WEIGHTS`
(and `BRAIN_QWEN_TOKENIZER`) at it -- a `.gguf` file works here too.

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

### LoRA adapters

A named LoRA adapter is stored beside its base checkpoint in the model store,
at `<models-dir>/Qwen/Qwen3-0.6B/adapters/<owner>/<name>/<tag>/`, and is
addressed as `Qwen/Qwen3-0.6B:<owner>:<name>:<tag>`. The tag defaults to
`latest` and is **overwritten on every rerun** of the same adapter name - a
finetune run always fully retrains and replaces that tag, it never
incrementally continues a previous run.

## Options

Which checkpoint/tokenizer to serve is still selected through env vars
(`BRAIN_QWEN_WEIGHTS`, `BRAIN_QWEN_TOKENIZER`, `BRAIN_FLUX2_TE`, above) -
everything about HOW to serve it is a `brain serve` flag:

- `--lora N` - LoRA rank for `finetune`.
- `--lora-targets LIST` - the projections a `finetune --lora` adapter
  covers, comma-separated from `wq,wk,wv,wo,gate,up,down` (the default is
  all seven). An unknown or repeated name is refused.
- `--base-dtype f32|bf16` - the storage dtype of the frozen base during a
  `finetune --lora` run (default `f32`). `bf16` halves the base's bytes - a
  7B decoder trains on one 24 GB card - while the adapters, activations
  and optimiser stay fp32. `--weights` may name a `transformers` checkpoint
  directory (a model-store `vendor/repo`): it is read as downloaded, one
  tensor at a time, and nothing is written beside it.
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
- `--device cpu|gpu|vulkan` - backend selection.
- `--qwen-ctx N` - built context length. Unset by default: the resident
  auto-sizes it to the target device's real usable VRAM instead of a fixed
  number, so raising the ceiling normally means freeing VRAM (or adding a
  card), not passing this flag. Set it to pin an exact value instead of the
  auto-picked one. Neither the auto-sized nor an explicit value is capped at
  the checkpoint's own trained context: going past it derives a YaRN
  (arXiv 2309.00071) RoPE scaling automatically (`factor = ctx / native`,
  the same ratio a real long-context checkpoint's own `rope_scaling` key
  would carry) - a checkpoint that already declares one is never
  second-guessed.
- `--qwen-max-batch N` - concurrent serving batch slots (default 16).
- `--qwen-kv-fp32` - opt out of the default int8 KV cache.
- `--qwen-kv-calib` - opt in to per-head KV clip ranges produced by `brain
  qwen3 calib` (off by default).
- `--qwen-kv-offload-gb N` - host RAM the serving engine may use to park
  preempted sessions' KV cache (default `0`, off). See below.
- `--qwen-weights-int8` / `--qwen-weights-fp32` - quantize the 7
  per-layer linears to int8, or keep them fp32. Unset, a checkpoint of 6B
  parameters or more is quantized (its fp32 weights do not fit one 24 GB
  card) and a smaller one is not - at the real Qwen3-8B config, int8
  shrinks the weight term from ~20.8 GiB to ~11.9 GiB. A device with no packed-int8 dot path (the CPU
  backend, or an unusual GPU) degrades to fp32 with a printed warning
  rather than failing.
- `--qwen-max-prefill N` - cap the chunked-prefill row count below its
  512 default (clamped to `1..=512`; can only shrink it). The scores/probs
  scratch buffer this bounds is the single largest per-token-scaling cost
  in the whole engine - larger than the KV pool itself - and shrinks
  LINEARLY with this value: 512→128 saves ~2.25 GiB at ctx=24576 on the
  real Qwen3-8B config, with no effect on decode throughput or the KV
  pool's own size (only the prefill chunk shape changes: more, smaller
  passes over the same prompt).

## Serving more sessions than the KV pool holds

The KV pool is sized for a couple of full-length contexts, while the batch
has `BRAIN_QWEN_MAX_BATCH` slots, so a busy server can admit more sessions
than the pool can keep cached at once. Without host offload that ends badly:
the pool runs dry mid-decode and the serving lane fails with
`KV pool exhausted`.

Give the engine host RAM and it preempts instead. A session the scheduler is
not advancing this round has its whole KV copied to host memory and its GPU
blocks handed to the sessions that are decoding; when its turn comes round
again the bytes are copied back and it continues **exactly** where it left
off - the restored cache is byte-identical, so the tokens it goes on to
produce are identical to the ones it would have produced had it never been
preempted. That is a tested property, not an aspiration.

```bash
brain serve --openai 8080 --qwen-kv-offload-gb 16
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

## Hardware and limits

The serving engine doesn't yet reuse a shared prompt prefix across
separate requests (each request's prefill is independent). Mixture-of-experts
style configs are defined in the parameter layout but the serving engine
currently serves dense configs only.

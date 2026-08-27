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

`Qwen/Qwen3-0.6B` is auto-fetched from Hugging Face on first use. You can
also import a checkpoint you already have locally:

```bash
brain qwen3 import --hf /path/to/Qwen3-0.6B --out qwen.safetensors
```

A **GGUF** works too -- llama.cpp's `qwen3` architecture, e.g.
`Qwen/Qwen3-8B-GGUF` -- through the generic importer, which picks the right
converter from the file's own `general.architecture`:

```bash
brain import /path/to/Qwen3-8B-Q8_0.gguf --out qwen3-8b.safetensors
```

Both routes produce the same brain parameters. The tensor-name map is
transcribed from llama.cpp's own `gguf-py/gguf/tensor_mapping.py` and
`constants.py`, and a test asserts the two routes agree *bit for bit* on the
same logical checkpoint -- not to a tolerance, because the mistake worth
catching (a swapped `k`/`v` projection) is shape-compatible on every GQA layer
and only an exact comparison sees it.

Note the disk trade: brain's own checkpoint format is fp32, so importing a
quantized GGUF *expands* it: a Q8_0 file is a much smaller download than the
bf16 safetensors, and the fp32 conversion is larger than either. Where a
component reads a GGUF directly -- FLUX.2's text encoder, below -- no
conversion happens and the download saving is the whole story.

To make a checkpoint a `brain serve` resident, point `BRAIN_QWEN_WEIGHTS`
(and `BRAIN_QWEN_TOKENIZER`) at it.

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

Other verbs: `brain qwen3 export` (to ONNX), `brain qwen3 precompile`
(precompile kernels for a target device), `brain qwen3 toolcall` (tool-call
evaluation).

### LoRA adapters

A named LoRA adapter is stored beside its base checkpoint in the model store,
at `<models-dir>/Qwen/Qwen3-0.6B/adapters/<owner>/<name>/<tag>/`, and is
addressed as `Qwen/Qwen3-0.6B:<owner>:<name>:<tag>`. The tag defaults to
`latest` and is **overwritten on every rerun** of the same adapter name - a
finetune run always fully retrains and replaces that tag, it never
incrementally continues a previous run.

## Options

- `--lora N` - LoRA rank for `finetune`.
- `--device cpu|gpu|vulkan` - backend selection.
- `BRAIN_QWEN_CTX` - built context length (default 24576).
- `BRAIN_QWEN_MAX_BATCH` - concurrent serving batch slots (default 16).
- `BRAIN_QWEN_KV_INT8` - int8 KV cache, on by default; `--kv-fp32` or
  `BRAIN_QWEN_KV_INT8=0` opts out.
- `BRAIN_QWEN_KV_CALIB` - per-head KV clip ranges produced by `brain qwen3
  calib` (unset by default).

## Hardware and limits

The LoRA finetune target set is fixed - the four attention projections plus
the three MLP projections - there's no `--targets` flag to narrow or widen
it yet. The serving engine doesn't yet reuse a shared prompt prefix across
separate requests (each request's prefill is independent). Mixture-of-experts
style configs are defined in the parameter layout but the serving engine
currently serves dense configs only.

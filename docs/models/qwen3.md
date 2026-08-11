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
| CLI (`brain do`)       | [x] |
| HTTP API               | [x] |
| D-Bus                  | [x] |
| Batched serving        | [x] |

## Getting the weights

`Qwen/Qwen3-0.6B` is auto-fetched from Hugging Face on first use. You can
also import a checkpoint you already have locally:

```bash
brain qwen import --hf /path/to/Qwen3-0.6B --out qwen.safetensors
```

To make a checkpoint a `brain serve` resident, point `BRAIN_QWEN_WEIGHTS`
(and `BRAIN_QWEN_TOKENIZER`) at it.

## Running it

```bash
# Inference
brain qwen infer --weights qwen.safetensors --prompt "Hello"

# Training from scratch
brain qwen train --data data/shakespeare_char --out qwen-train.safetensors --steps 1000

# LoRA finetune -- trains a NAMED adapter
brain qwen finetune --lora 8 --weights Qwen/Qwen3-0.6B \
    --adapter my-org/my-finetune --dataset /path/to/dataset/dir --steps 500

# Prove it learned: held-out loss/accuracy, base vs. the adapter
brain qwen eval --weights Qwen/Qwen3-0.6B \
    --adapter my-org/my-finetune --jsonl /path/to/validation.jsonl

# Full-parameter finetune from a plain checkpoint file (no adapter)
brain qwen finetune data/shakespeare_char --weights qwen.safetensors \
    --out qwen-ft.safetensors --steps 500

# Concurrent serving (paged KV, continuous batching)
brain qwen serve --weights qwen.safetensors --port 8080

# Device selection
brain qwen infer --weights … --device cpu     # JIT-compiled CPU path
brain qwen infer --weights … --device gpu     # portable GPU backend (default)
brain qwen infer --weights … --device vulkan  # native Vulkan
```

Other verbs: `brain qwen export` (to ONNX), `brain qwen precompile`
(precompile kernels for a target device), `brain qwen toolcall` (tool-call
evaluation).

### LoRA adapters

A named LoRA adapter is stored beside its base checkpoint in the model store,
at `<models-dir>/Qwen/Qwen3-0.6B/adapters/<owner>/<name>/<tag>/`, and is
addressed as `Qwen/Qwen3-0.6B:<owner>:<name>:<tag>`. The tag defaults to
`latest` and is **overwritten on every rerun** of the same adapter name — a
finetune run always fully retrains and replaces that tag, it never
incrementally continues a previous run.

## Options

- `--lora N` — LoRA rank for `finetune`.
- `--device cpu|gpu|vulkan` — backend selection.
- `BRAIN_QWEN_CTX` — built context length (default 24576).
- `BRAIN_QWEN_MAX_BATCH` — concurrent serving batch slots (default 16).
- `BRAIN_QWEN_KV_INT8` — int8 KV cache, on by default; `--kv-fp32` or
  `BRAIN_QWEN_KV_INT8=0` opts out.
- `BRAIN_QWEN_KV_CALIB` — per-head KV clip ranges produced by `brain qwen
  calib` (unset by default).

## Hardware and limits

The LoRA finetune target set is fixed — the four attention projections plus
the three MLP projections — there's no `--targets` flag to narrow or widen
it yet. The serving engine doesn't yet reuse a shared prompt prefix across
separate requests (each request's prefill is independent). Mixture-of-experts
style configs are defined in the parameter layout but the serving engine
currently serves dense configs only.

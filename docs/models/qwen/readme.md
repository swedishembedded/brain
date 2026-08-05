# Qwen3 decoder in brain (`crates/qwen`)

Qwen3 dense decoder Transformer — GQA + QK-norm + RoPE + SwiGLU — running on
brain's fp32/WGSL engine. Supports training from scratch, HuggingFace weight
import, LoRA finetuning, INT8 weight quantization, tensor/expert sharding, and
a **concurrent paged-KV serving engine** with continuous batching, int8 KV
cache, speculative decoding, and a device-side greedy head.

`docs/models/qwen/status.md` is the workstream ledger (what landed, parity
numbers, every trap the gates caught). This file is the user-facing guide.

## Quick start

```bash
make release

# Import from HuggingFace safetensors
./target/release/brain qwen import --hf /path/to/Qwen3-0.6B --out qwen.safetensors

# Inference
./target/release/brain qwen infer --weights qwen.safetensors --prompt "Hello"

# Training
./target/release/brain qwen train --data data/shakespeare_char --out qwen-train.safetensors --steps 1000

# LoRA finetune -- trains a NAMED adapter, saved beside its base in the
# model store (<models-dir>/Qwen/Qwen3-0.6B/adapters/<owner>/<name>/<tag>/),
# addressable as Qwen/Qwen3-0.6B:<owner>:<name>:<tag> (tag defaults "latest",
# and is OVERWRITTEN on every rerun -- "fully retrain and overwrite").
./target/release/brain qwen finetune --lora 8 --weights Qwen/Qwen3-0.6B \
    --adapter my-org/my-finetune --dataset /path/to/bench/generic-messages-v2/dir \
    --steps 500
# or: make train/qwen/lora DATASET=<dir> ADAPTER=my-org/my-finetune

# Prove it learned: held-out loss/accuracy, base vs the adapter, on the same
# bench-exported validation split (see docs/guides/training.md).
./target/release/brain qwen eval --weights Qwen/Qwen3-0.6B \
    --adapter my-org/my-finetune --jsonl /path/to/validation.jsonl

# Full-parameter finetune from a plain checkpoint file (no --lora, no store
# ref needed) -- the older, still-supported path.
./target/release/brain qwen finetune data/shakespeare_char --weights qwen.safetensors --out qwen-ft.safetensors --steps 500

# Concurrent serving (paged KV, continuous batching)
./target/release/brain qwen serve --weights qwen.safetensors --port 8080

# Device selection
brain qwen infer --weights … --device cpu     # WGSL→Cranelift JIT
brain qwen infer --weights … --device gpu     # wgpu (default)
brain qwen infer --weights … --device vulkan  # native Vulkan
```

## Architecture

Pre-norm decoder, RMSNorm throughout, untied `lm_head`, masked CE. Per layer:

- **GQA** (Grouped Query Attention): `n_kv_heads < n_heads` with per-head
  QK-RMSNorm applied to q/k before RoPE.
- **RoPE** (Rotary Position Embeddings): base 1e6, half-split layout.
- **SwiGLU MLP**: gated activation with no attention/MLP biases.
- **Decoupled `head_dim`**: e.g. hidden 1024 but 16 heads × 128 = 2048 ≠ 1024.

### Serving engine (`serve.rs`)

The paged-KV serving engine is the central serving workstream. Key
features:

| Feature | What it does |
|---|---|
| Paged KV cache | Shared block pools, no per-sequence worst-case reservation |
| Batched ragged decode | One forward per iteration serves all active sequences |
| Chunked prefill | Long prompts split into fixed-size chunks for bounded latency |
| Int8 paged KV | ~4× smaller KV pool via on-read dequantization |
| Speculative decoding | Draft-then-verify for higher throughput |
| Device-side greedy head | `ARGMAX_ROW/PART/FINAL` — decode never ships `[batch, vocab]` to host |
| Decode-regime kernels | Per-dispatch kernel selection by row count (`RMSNORM_ROWS`, `MATMUL_GEMV`) |
| Int8 weight path (A0) | Per-token activation quant + DP4A GEMMs with per-token × per-channel dequant scales |
| On-device decode window (A4) | Feed argmax back as next input without host round-trip |
| Continuous batching | Multi-sequence concurrent decode with queue-age-aware scheduling |

### How it is built (brain mapping)

| Piece | Shared implementation |
|---|---|
| RMSNorm | `model::block::rmsnorm_fwd` (coalesced `_rows` kernel) |
| RoPE | `model::block::rope_fwd` (paged variant for serving) |
| Attention | `model::block` GQA kernels (paged/ragged-batched for serving) |
| SwiGLU | `silu_mul` kernel |
| Linear projections | `matmul` / `matmul_rows` / `matmul_gemv` (regime-selected) |
| INT8 weights | `model::int8` (DP4A GEMMs, per-channel symmetric) |

## CLI

```
brain qwen import       # HF safetensors → brain .safetensors
brain qwen infer        # interactive inference
brain qwen serve        # concurrent paged-KV serving
brain qwen export       # export to ONNX
brain qwen precompile   # precompile kernels for a target device
brain qwen train        # training from scratch
brain qwen finetune     # full-parameter OR named-LoRA-adapter finetuning
brain qwen eval         # held-out loss/accuracy, base vs a named adapter
brain qwen toolcall     # tool-call evaluation
```

## Makefile targets

```bash
make data/gpt                             # generate training data
make train/gpt/qwen                       # train Qwen3
make eval/gpt/qwen                        # evaluate perplexity
```

## Limitations

- `brain qwen finetune --lora`'s target set is fixed (the four attention
  projections plus the three MLP projections, `wq`/`wk`/`wv`/`wo`/`gate`/`up`/
  `down`) -- not yet a `--targets` flag.
- The serving engine does not yet support prefix caching across requests
  (the `PrefixCache` infrastructure exists in `model::paged` but is not yet
  adopted in `serve.rs`).
- Expert sharding for MoE-style configs is defined in the param layout but the
  serving engine serves dense configs only.

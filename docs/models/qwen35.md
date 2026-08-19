# Qwen3.8-27B (dense hybrid GDN/GQA decoder + MTP + vision)

A 64-layer dense hybrid decoder - the dense sibling of
[`qwen35moe`](qwen35moe.md): 3:1 Gated DeltaNet (chunked linear-attention) to
GQA layers, a sigmoid attention-output gate, partial RoPE + M-RoPE on the GQA
layers, but a plain dense SwiGLU MLP on every layer instead of a sparse MoE.
Adds a single-layer multi-token-prediction (MTP) head sharing the token
embedding and LM head, and a spliced Qwen3-VL-style vision tower (ViT +
PatchMerger, reused unchanged) for image input. `reasoning_effort` is a
chat-template concept, not an architectural one - this port has no verified
prompt-injection convention for it yet (see below).

## Support

| Capability | Supported |
|---|---|
| Inference             | [x] |
| Training from scratch | [x] |
| LoRA fine-tune         | [x] (rank/alpha adapters on all 12 targetable GDN/GQA/MLP projections) |
| MTP head               | [x] (structural - no reference oracle available, gradchecked and overfit-tested only) |
| Vision input            | [x] (real-dims parity-tested tower; end-to-end splice is structural only - no reference oracle for the composite) |
| CLI                    | [x] |
| HTTP API               | [x] |
| D-Bus                  | [x] |
| Batched serving        | [x] (paged, single-GPU, one active sequence at a time on the GPU - see "Hardware and limits") |

## Getting the weights

Model id: `brain/qwen35`, upstream `Qwen/Qwen3.8-27B-FP8`. Weights ship as
DeepSeek-V3-style blockwise FP8 (E4M3 + BF16 `weight_scale_inv`), dequantized
host-side at import. No GGUF importer for this architecture - the real
checkpoint is safetensors-only.

## Running it

```bash
brain qwen35 infer --weights qwen35.safetensors --tokenizer tokenizer.json --prompt "..."
```

Serving (HTTP/D-Bus, paged continuous batching):

```bash
BRAIN_QWEN35_WEIGHTS=qwen35.safetensors BRAIN_QWEN35_TOKENIZER=tokenizer.json brain serve
```

## Hardware and limits

The serving engine is single-GPU, fp32 weights and fp32 KV cache only (no
int8 weight or KV tier for this architecture yet), and processes one truly
active sequence at a time on the GPU - several sequences may be resident and
interleaved by the scheduler across iterations, but never batched together
into one GPU dispatch. Prefill replays the prompt one token at a time rather
than a fast batched forward. No LoRA adapter folding into the serving path
yet (the adapters train and gradient-check correctly; loading a trained
adapter into `brain qwen35 infer`/serving is not wired). No multi-GPU
sharding wired into serving (the underlying `model::Shardable` capability
exists and is gated by its own test, `crates/qwen35/tests/shard_parity.rs`,
which self-skips without 2+ discrete GPUs - not available on every box).
`reasoning_effort` (xhigh/medium/low) is not implemented: no verified
Qwen3.8 prompt-injection convention was found to build it against without
guessing - only the existing `enable_thinking` boolean is wired, reused from
`qwen3::chat`.

# GPT decoder in brain (`crates/gpt`)

Dense GPT decoder Transformer (nanoGPT parity): forward + backprop as WGSL
compute dispatches. It is the from-scratch baseline the rest of the engine
(MoE / Qwen / GLM) is measured against, and the reference architecture behind
the `brain bench` learnability suite.

## Architecture

Pre-norm decoder, LayerNorm with bias throughout, untied `lm_head`, masked
cross-entropy. Per layer (`crates/gpt/src/model.rs`):

- **Embeddings**: `x = tok_emb[idx] + pos_emb[pos]` — token embedding +
  **learned absolute** positional embedding (no RoPE).
- **LayerNorm** (LN1/LN2 per block + final `ln`), **pre-LN** placement,
  dispatched through the shared `model::block` LayerNorm family
  (`layernorm_fwd` / `ln_stats_fwd` / `layernorm_dx_bwd`).
- **Attention**: causal multi-head attention (MHA, `n_heads == n_kv_heads`),
  fused `qkv` projection (biased) → scores → softmax → apply → `attn.out`
  (biased). Causality is baked into the attention WGSL kernels.
- **MLP**: `mlp.fc` (d→d_ff) → **GELU** (tanh approximation) → `mlp.proj`
  (d_ff→d), each with bias. `d_ff` defaults to `4 * d_model`.
- **Head**: **untied** `lm_head` (no bias) → logits.
- **Loss**: masked cross-entropy, `ignore_index = IGNORE`, normalized by the
  non-ignored position count, so masked datasets work.
- **Init** (`init.rs`): GPT-2-style — `Normal(0, 0.02)`, residual projections
  scaled by `0.02 / sqrt(2 * n_layers)`, LN gain 1 / bias 0.

`GptConfig` (`vocab, block_size, n_layers, d_model, n_heads, d_ff`) round-trips
as JSON with `"model": "gpt"`.

## CLI

```
brain gpt train <data_dir> [--out F --steps N --batch B --block T --lr X
                            --layers L --d-model D --heads H --warmup N
                            --grad-accum N --seed S --mask <char> --align]
brain gpt eval  --weights F --data <dir> [--batches N --samples M --seed S]
brain gpt gen   --weights F [--data <dir> --prompt "..." --max-new N
                            --temp X --top-k K --seed S]
```

`gen` / `sample` / `generate` are all canonicalized to `infer`. Generation uses
the **incremental KV-cache** path (`generate_kv`); the vocab is read from the
checkpoint's embedded char-tokenizer `itos` (or `--data/meta.json` for BPE).

> **Bare `train` / `eval` / `generate` (no `gpt`) are the Sparse MoE model**,
> not GPT — see `docs/models/moe/readme.md`. The dense decoder is always
> `brain gpt …`.

`--device cpu|gpu|gpuN` (or `BRAIN_DEVICE`) selects the backend globally.

## What's implemented

- **Forward + backward** as pre-recorded WGSL step lists + an AdamW optimiser
  step.
- **Training** (`train.rs`) — delegates to the shared `model::train::fit`
  (cosine LR + warmup, grad-accum, grad-clip, periodic eval, resumable
  atomic checkpointing).
- **Sampling** (`sample.rs`) — `generate` (O(T²) full recompute) and
  `generate_kv` (O(T) incremental KV-cache); temperature + top-k, greedy
  argmax when `temp ≤ 0`.
- **Incremental KV-cache decode** — persistent per-layer K/V cache, lazy-built,
  single-token `step()`.
- **Checkpoint save/load** — brain `.safetensors` with an embedded
  char-tokenizer `itos`; streaming mmap load.
- **Pipeline sharding** (`shard.rs`) — implements `model::Shardable` for
  `model::Pipeline` multi-GPU pipeline parallelism (the untied head needs no
  replicated params).

**Not in this crate** (vs Qwen): no HF import, no INT8 quant, no paged-KV
serving engine — those live in `crates/qwen`. GPT reads/writes brain-format
checkpoints only.

## Parity / gradcheck

- `gradcheck::check_gpt` — finite-difference gradient check (central diff,
  `eps=5e-3`) over a tiny config; the `gpt_analytic_grads_match_finite_differences`
  test asserts no failures at `atol=4e-3, rtol=8e-2`.
- `kv_step_matches_full_recompute` — incremental cache vs O(T²) recompute,
  `max_abs < 2e-3`.
- `streaming_load_matches_eager` — mmap load byte-identical to the eager path.
- **Learnability** (`tests/convergence.rs`): memorize cyclic (`< 0.10`),
  copy-through-mask (`< 0.90` vs marginal ln4≈1.386), reverse-through-mask
  (`< 1.30` vs ln5≈1.609), loss improves with model size.
- **Multi-GPU parity**: `shard_forward_and_grad_parity_gpt` (2-stage pipeline,
  loss rel `< 1e-4`, grads rel `< 1e-3`); `dp_grad_parity_gpt`
  (`DataParallel` over 2 GPUs, grads rel `< 1e-3`).
- **CPU register GEMM** — `cpu_register_equals_cpu_naive` (`max_abs < 1e-4`).

## Bench

Registered in `arch_registry()` (`crates/bench/src/arch.rs`): `gpt` (baseline,
size per-benchmark), `gpt-small` (L1×D32×H2), `gpt-wide` (L2×D96×H4). Run with
`make bench/eval ARCH=gpt` / `make bench/compare`.

## Kernel / block reuse

LayerNorm goes through the shared `model::block` helpers; every other kernel
comes from the shared `brain-kernels` crate. GPT declares no local WGSL and
re-implements nothing — it uses LayerNorm (not RMSNorm), no RoPE, and GELU
(not SwiGLU), so none of the RMSNorm/RoPE/SwiGLU family it doesn't use is
duplicated here.

## Limitations

- `lm_head` is **untied**, not tied to `tok.weight` as in nanoGPT — tying (grad
  accumulation into the embedding) is an explicit follow-up.
- GELU uses the **tanh approximation**, not exact erf.
- No HF import, no INT8, no paged-KV serving (those are Qwen's).
- `model::step()` (KV decode) requires a whole-model shard.

## See also

- `README.md` — the toy char-level task, `make train/gpt/<name>`, honest-eval
  methodology (§3).
- `docs/testing.md` — the gradcheck gate and GPT test coverage.
- `docs/architecture.md` — crate graph.
- `AGENTS.md` → Models → GPT decoder.

# Sparse MoE Transformer in brain (`crates/moe`, `crates/federated`)

A Sparse Mixture-of-Experts Transformer (RMSNorm / RoPE, top-k routed SwiGLU
experts, tied head) for studying **memorization vs. generalization** on a toy
64-symbol next-token rule with exact ground truth — the canonical brain task
(`README.md`). Paired with **vertical expert sharding** for federated training
(`crates/federated`).

## The toy task

A 64-symbol vocabulary and a next-token rule
`data[i] = (data[i-2] + table[data[i-1]]) % vocab`, where `table` is a random
permutation of `0..vocab` (a substitution table); a random reset token every
257th position; corpus size 20,000. A model that has *memorized* `table`
reproduces the orbit; one that has *generalized* the rule continues a **NEW
orbit** (reset-free, from an unseen starting pair).

Eval (`run_eval`) reports train-orbit, val-orbit, and NEW-orbit accuracy,
sweeping context lengths `[2, 8, 16, 32, block_size]`, with the random baseline
`100/vocab %` printed alongside — so memorization and generalization are
separable, not conflated. See `README.md` for the honest-eval methodology.

## Architecture

Pre-norm decoder, RMSNorm throughout, tied `lm_head`, masked cross-entropy.
Per layer (`crates/moe/src/model.rs`, `train.rs`):

- **RMSNorm** (pre-norm): `norm1` (pre-attn), `norm2` (pre-MoE), final `norm`.
- **RoPE** — applied to the Q and K regions of the fused QKV buffer; **interleaved**
  (adjacent-pair) layout, base 10000. (Not the half-split layout Qwen3/GLM use.)
- **Attention**: causal MHA, fused QKV → RoPE → attention. Inference uses one
  fused causal `ATTENTION` kernel (online softmax); training splits into
  scores / softmax / apply (each with its backward).
- **Router**: top-k-of-`n_experts` — `router.weight` matmul → logits →
  `ROUTER_GATE` (top-k selection + renormalized gate). **Dense top-k**: all
  experts are evaluated and masked by the renormalized gate; capacity dropping
  is disabled. Default `n_experts=4, top_k=2` → **top-2-of-4**.
- **Experts (SwiGLU)**: per expert `w_gate` / `w_up` / `w_down`;
  `h = silu(gate(x)) * up(x)` then `w_down(h)`, each scaled by its gate weight
  and accumulated (`SCALE_ADD`).
- **Head**: **tied** `lm_head` (reuses `token_emb.weight`).
- **Loss**: masked cross-entropy (`ignore_index = IGNORE`) + a load-balance aux
  loss + a router z-loss folded into the router gradient
  (`aux_coef=0.01, z_coef=1e-4`).
- **Optimizer**: AdamW (betas `0.9/0.95`) via the shared `optim::Optim`, with
  grad clip + grad-accum.

Config defaults: `vocab=64, n_layers=2, d_model=64, n_heads=4, n_experts=4,
top_k=2, d_ff=128`. fp32, WGSL compute, `@workgroup_size(64)`.

## CLI

There is **no `brain moe` subcommand** — the bare verbs are the MoE model
(`crates/cli/src/main.rs`):

```
brain train   [--steps N --batch-size B --block-size T --lr X --weight-decay X --seed S --out F]
brain eval    --weights F [--seed S --samples N]
brain generate --weights F [--prompt 1,2,3,4 --max-new N --temperature X --top-k K --seed S]
```

`brain train` writes `moe_rs.safetensors`; `brain eval` runs the
train/val/NEW-orbit sweep; `brain generate` samples (temperature + top-k;
`DUMP_LOGITS` dumps raw logits). For the dense GPT decoder, use `brain gpt …`
(see `docs/models/gpt/readme.md`).

### Federated / sharded training (`brain federated …`)

```
brain federated split    <base.safetensors> <out_dir>
brain federated verify   <dir>
brain federated merge    <dir> --out <full.safetensors>
brain federated assemble <base_dir> [overlay_dir ...] --out <full.safetensors>
brain federated train-expert --base B --expert E --out DIR [--steps N --batch B --block T --lr X --seed S]
```

`make federated-demo` runs the round trip: `train` → `split` → `verify` →
`merge`. The full lifecycle and what remains are in `docs/federated.md`.

**Vertical expert split** — expert `E` spans *every* layer
(`blocks.<L>.moe.experts.<E>.{w_gate,w_up,w_down}`); the shared backbone is
embeddings, attention, norms, `router.weight`, and the head. A checkpoint dir
is `shared.safetensors` + `experts/expert_NNNN.safetensors` + `manifest.json`
(per-file SHA-256 + a base-config hash). `verify` re-hashes every file and
confirms the shared config hash matches the manifest base hash (rejects
tampering / a wrong base). `assemble` layers base + overlays **last-wins** per
expert id, requiring all shards to share the base config hash.

`train-expert` is the federated worker step: `freeze_grads_except_expert(e)`
zeros non-expert grads, and with `wd=0.0` the backbone + other experts stay
**bit-for-bit unchanged** — so each shard trains independently and
auditably on `base.safetensors` alone.

## What's implemented

Forward + backward (WGSL, SSA-style activation cache); training from scratch
(via the shared `model::train::fit` through the `model::Model` impl); generation
(temperature + top-k sampling); eval (the train/val/NEW-orbit sweep); and the
full checkpoint-level federation pipeline (split/verify/merge/assemble/
train-expert). **No HF import** (MoE trains from scratch on its own task); **no
serving engine** (one-shot CLI generation).

### Parity / gradcheck

- `gradcheck::check_moe` — finite-difference gradient check (central diff,
  `eps=5e-3`) over a tiny config; `moe_analytic_grads_match_finite_differences`
  asserts no failures at `atol=4e-3, rtol=8e-2`. The tiny config uses
  `top_k == n_experts` so the router is smooth (the hard top-k selection
  boundary itself is not FD-verifiable).
- `train_scope_freezes_backbone_and_trains_one_expert` — 30 steps on expert 1:
  backbone + expert 0 unchanged, expert 1 moved.
- `split_assemble_roundtrip_is_identity` — split → assemble is tensor-for-tensor
  identity and `verify` passes; `overlay_replaces_one_expert_last_wins` /
  `assemble_last_wins_across_three_files` pin the overlay semantics.
- `sha256::known_vectors` — empty/abc/fox SHA-256 vectors.

## Kernel / block reuse

Every kernel is a shared `brain-kernels` string (RMSNorm, RoPE, attention,
`ROUTER_GATE`, `SILU_MUL`, `SCALE_ADD`, matmuls + their backwards, AdamW,
grad-norm/clip) — MoE declares no local WGSL and re-implements nothing
(no local rmsnorm/rope/silu/matmul). Shared infra: `gpu_core`, `paramstore`,
`optim`, `checkpoint`, `data::binio`, `model::train::fit`.

## Bench

Registered in `arch_registry()` as `moe` (`MoeDecoder` implements the bench
`Decoder` trait). `brain bench eval --arch moe` runs the whole learnability
battery against MoE, directly comparable to `gpt` / `qwen` / `glm`.

## Limitations

- **No HF import** — trains from scratch on the 64-symbol toy task only (MoE on
  the shared char datasets is a documented follow-up).
- **No serving engine** (one-shot CLI generation).
- **No true memory sharding** — the whole model is GPU-resident during a
  federated worker run; CPU offload / layer-expert shards are not implemented
  (`docs/federated.md`). What you get today is independent, sequential, auditable
  per-expert training.
- Router-only integration pass, anchor-KL + router-selectivity losses, and
  marginal-utility / ablation eval are not wired (`docs/federated.md`).
- fp32 only; dense top-k evaluates all experts (no sparse dispatch / capacity
  dropping).

## See also

- `README.md` — the toy task and the honest-eval methodology.
- `docs/federated.md` — the federated pipeline (done vs remaining).
- `docs/federated-moe.md` — the long-form source-design essay.
- `docs/models/gpt/readme.md` — the dense baseline (`brain gpt …`).
- `AGENTS.md` → Models → Sparse MoE Transformer.

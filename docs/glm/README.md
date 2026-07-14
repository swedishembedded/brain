# GLM-5.2 (`glm_moe_dsa`) in brain

A from-scratch GLM-5.2 decoder on brain's fp32/WGSL engine — train / evaluate /
infer a tiny GLM, or import official HuggingFace weights — running on cpu/gpu
(and an NPU export path). Reference material: `resources/glm/` (exact configs,
HF/vLLM/SGLang source, papers).

## Architecture (crate `glm`)

Pre-norm decoder, RMSNorm throughout, untied `lm_head`, masked CE. Per layer:

- **MLA** (Multi-head Latent Attention): low-rank q (`q_a`→RMSNorm→`q_b`) and kv
  (`kv_a_c`→RMSNorm→`kv_b`), a decoupled **nope/rope head split**, **interleaved
  RoPE** on the rope slice, and a shared **MQA** rope key. brain splits the fused
  HF projections into contiguous buffers (`q_b_nope`/`q_b_rope`, `kv_b_nope`/
  `kv_b_v`, `kv_a_c`/`kv_a_rope`) so every RoPE/matmul target stays contiguous.
- **MoE**: a sigmoid **`noaux_tc`** router (per-expert selection bias, group-
  limited top-k, renorm, `routed_scaling_factor`), a **shared** always-on expert,
  and a `first_k_dense_replace` **dense→MoE** layer schedule. The selection bias
  is not backprop'd (Frozen), matching the reference.
- **DSA indexer** (optional, `indexer_full` schedule): a detached, forward-only
  side-network that selects the top-`index_topk` keys per query, with cross-layer
  **IndexShare** (`Full` layers compute; `Shared` reuse). At `index_topk >= seq`
  it is all-pass ⇒ exactly dense MLA.
- **MTP** (optional, `mtp`): a lightweight position-wise head predicting token
  t+2 (aux CE loss + speculative-draft head).

`config.rs` has `tiny()` (tests) and `glm5_2()` (published shape) presets.

## Status (what's validated)

| Area | State | Evidence |
|---|---|---|
| MLA + MoE core (fwd/bwd) | ✅ | `gradcheck::check_glm` (all params vs finite-diff) |
| Learnability | ✅ | `crates/glm/tests/convergence.rs` (overfit, cyclic memorize, scaling) |
| DSA indexer + IndexShare | ✅ | `crates/glm/tests/indexer.rs` (all-pass≡dense, sparse-restricts, trains) |
| MTP head | ✅ | `gradcheck::check_glm_mtp` + learnability |
| cpu / gpu | ✅ | end-to-end `brain glm train/eval/infer` (shared `Gpu`/`Step` seam) |
| HF import (single/sharded) | ✅ | name-map + de-interleave + packed-expert unit tests |
| bench arch `glm` | ✅ | `brain bench eval --arch glm` |
| NPU fp32 ONNX export | ✅ (validated on HW) | `crates/npu/tests/glm_onnx.rs`: parity vs brain on OpenVINO **CPU + Intel NPU** (`max_abs ≈ 0.005`, argmax agrees). `docs/glm/NPU.md` |

## CLI

`brain glm <train|finetune|infer|eval|import|export>` (shared arg grammar with
`gpt`/`qwen`; `--size tiny|small|base` presets). Import: `--hf <dir>` (single or
sharded safetensors). Export: `--out model.onnx --seq T`.

## Remaining: DSA indexer distillation training (implementation plan)

The indexer is **detached from the LM loss** — it works today via imported (real,
trained) weights or in all-pass dense mode, but training it *from scratch* for
genuine long-context sparsity needs a separate distillation objective (as in
DeepSeek-V3.2 / the IndexCache paper — the reference also gates this behind a
dedicated pipeline). Not yet implemented (no validation oracle in a from-scratch
setting; ~8 new backward kernels). Concrete plan:

1. **Target**: the dense MLA attention distribution over keys per query, averaged
   over heads (`p_target[b,s,t] = mean_h probs[b,h,s,t]`, already computed in the
   forward for `Full` layers when run dense).
2. **Loss**: `KL(softmax(index_scores) ‖ p_target)` (or top-k-recall / MSE) per
   query, over `Full` indexer layers — a separate scalar, added to training with
   its own weight, differentiated **only** w.r.t. the `idx.*` params.
3. **Backward kernels** (new, forward-only today): grads through `mla_index_scores`
   (relu + per-head weighting) → `wq_b`, `wk`, `weights_proj`, the indexer
   LayerNorm, and `rope_sub`. Mirror the MLA backward pattern.
4. **Optimizer**: the `idx.*` params become a second `ParamStore`/`Optim` target
   (they are currently `Role::Frozen` for the CE path).
5. **Test**: the indexer's top-k recall vs dense attention improves over steps on
   a synthetic long-context recall task (`index_topk < seq`).

Until then, use `index_topk >= block_size` (dense, the default for tiny/from-
scratch) or import a checkpoint whose indexer is already trained.

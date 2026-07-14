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
| DSA indexer distillation training | ✅ | `Glm::distill_step` (host-side, RMS-normalized); grads FD-checked + convergence-tested (`glm::distill`), integration in `indexer.rs` |
| MTP head | ✅ | `gradcheck::check_glm_mtp` + learnability |
| cpu / gpu | ✅ | end-to-end `brain glm train/eval/infer` (shared `Gpu`/`Step` seam) |
| HF import (single/sharded) | ✅ | name-map + de-interleave + packed-expert unit tests |
| bench arch `glm` | ✅ | `brain bench eval --arch glm` |
| NPU fp32 ONNX export | ✅ (validated on HW) | `crates/npu/tests/glm_onnx.rs`: parity vs brain on OpenVINO **CPU + Intel NPU** (`max_abs ≈ 0.005`, argmax agrees). `docs/glm/NPU.md` |

## CLI

`brain glm <train|finetune|infer|eval|import|export>` (shared arg grammar with
`gpt`/`qwen`; `--size tiny|small|base` presets). Import: `--hf <dir>` (single or
sharded safetensors). Export: `--out model.onnx --seq T`.

## DSA indexer distillation training

The indexer is **detached from the LM loss**, so it is trained by a separate
objective — match the dense MLA attention distribution over keys (the
DeepSeek-V3.2 / IndexCache recipe). `Glm::distill_step(lr)` implements this
host-side (`crates/glm/src/distill.rs`): run a dense forward
(`index_topk >= block_size` ⇒ the cached `probs` are the unmasked attention),
and for every `Full` layer take a softmax-cross-entropy step
(`d score = softmax(index_scores) − mean_h probs`) that updates **only** the
`idx.*` params via a per-tensor **RMS-normalized** update (robust to the
indexer's tiny near-zero init). Validation: the backward is finite-difference
gradient-checked, RMS-GD convergence on a controlled peaked target is tested
(`glm::distill` unit tests), and the model integration (updates the indexer,
leaves the backbone frozen) is covered in `crates/glm/tests/indexer.rs`.

Typical use: train the backbone, then run `distill_step` for a while (the target
is fixed since the attention doesn't depend on `idx.*`), then deploy with
`index_topk < seq` for real sparsity. At random init both the attention and the
indexer are ~uniform, so there is nothing to distill until the backbone is
trained.

Everything the plan set out is now implemented and validated on cpu/gpu/npu.

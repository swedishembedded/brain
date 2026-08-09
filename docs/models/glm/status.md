# glm — workstream ledger

GLM-5.2 (`glm_moe_dsa`) decoder — MLA + sigmoid `noaux_tc` MoE + DSA indexer +
MTP — on brain's fp32/WGSL engine. The architecture and NPU design live in
`readme.md` / `npu.md` (not duplicated here). This file is the workstream
ledger — what landed, the parity gates, what remains.

## Done

- **MLA + MoE core (fwd/bwd)** — `model.rs` (`build_forward` / `build_backward`).
- **Learnability** — `crates/glm/tests/convergence.rs`: overfit, cyclic
  memorize, capacity scaling, MTP overfit.
- **DSA indexer + IndexShare** (forward-only) — `IdxMode::{Full,Shared,None}`;
  `crates/glm/tests/indexer.rs` (all-pass ≡ dense, sparse restricts attention,
  the model still learns).
- **Indexer distillation training** — `distill.rs` (`layer_distill`) +
  `model.rs::distill_step` (host-side, per-tensor RMS-normalized update);
  FD-checked + convergence-tested; integration covered in `indexer.rs`.
- **MTP head** (t+2 aux CE) — `gradcheck::check_glm_mtp` + learnability.
- **cpu / gpu + fast GEMM** — size-adaptive `matmul_reg2`; `reg2_parity.rs`.
- **HF import (single/sharded)** — `import.rs`; name-map + de-interleave +
  packed-expert fan-out tests.
- **bench arch `glm`** — `brain bench eval --arch glm`.
- **Incremental KV-cache decode** — `model.rs::step` / `decode_at` /
  `reset_cache`; `kv_step_matches_full_recompute`.
- **NPU fp32 ONNX export** — `npu/src/glm_topology.rs` +
  `glm_export.rs::build_glm_fp32_bytes`; `npu/tests/glm_onnx.rs`.
- **NPU INT8 export + NPU infer** — `glm_export.rs::build_glm_int8_bytes` /
  `export_glm_int8`; `glm_decode.rs::generate`; `--int8` + the `--device npu`
  CLI path; `glm_onnx_int8_runs` (gated).
- **Row-compacted MoE forward for inference** (2026-08-09) —
  `Glm::logits_all_compact`/`forward_compact` (`model.rs`), a parallel
  inference-only forward using `model::moe::expert_fwd_compact` instead of
  the dense per-expert loop; `sample::generate` now calls it. Bit-identical
  to `logits_all` (`compact_forward_tests::
  logits_all_compact_matches_logits_all`, mutation-verified at a tight
  `1e-5` tolerance — a real staleness bug this test caught along the way is
  documented in `.todo/glm-model-rs-compact-moe-wiring.md`), on both CPU
  and real Vulkan/P40 hardware. `build_forward`/`build_backward` (the
  training graph) are untouched — see the "Remaining" section below for
  exactly what this does and does not close.

## Parity ladder

| Gate | What | Result |
|---|---|---|
| `check_glm` | analytic grads vs central-FD over all trainable params (MLA, `noaux_tc` router, shared expert, dense→MoE, untied head); router bias Frozen | passes `atol=4e-3, rtol=8e-2` |
| `check_glm_mtp` | same FD check + the MTP t+2 path | passes `atol=4e-3, rtol=8e-2` |
| convergence | overfit `after < before*0.5`; cyclic `< 0.20`; MTP `after < before*0.6`; capacity `large < small + 0.10` | passes |
| indexer | all-pass ≡ dense (`< 1e-4`); sparse restricts (`> 1e-5`); IndexShare trains (`after < before*0.6`); distill moves idx weights, leaves the backbone unchanged | passes |
| distill math | idx grads vs FD `worst < 8e-2` (rel); RMS-GD converges `after < before*0.7` | passes |
| reg2 GEMM | `matmul_reg2` vs naive (same GPU) `< 1e-4`; GPU vs CPU `< 5e-3` | passes |
| KV decode | `step` vs recompute: hidden `< 3e-3`, logits `< 3e-3` | passes |
| HF import | full `param_list` coverage; packed-expert fan-out un-swapped/un-aliased | passes |
| NPU fp32 parity | ONNX vs brain forward: **argmax agrees per position**; `max_abs < 1e-2` (OpenVINO CPU), `max_abs < 2e-2` (Intel NPU w/ fallback); docs quote observed `max_abs ≈ 0.005` | passes |

## Remaining

- **GLM serving contract** — `docs/serving-contract.md` has no GLM entry and
  the OpenAI/Anthropic HTTP APIs do not yet cover `glm`. Real deferral: GLM is
  not yet discoverable/scheduled/batched/driven over D-Bus like yolo/z-image/asr.
- **Indexer backward is forward-only by design** — `idx.*` params are `Frozen`,
  excluded from the optimiser/gradcheck set, trained solely by `distill_step`
  detached from the LM loss. Not a deferral.
- **DSA sparse selection + MTP are not exported to the NPU** — the NPU runs
  dense attention (`index_topk >= seq`); MTP is a host-side draft loop. By
  design (see `npu.md`).
- **`npu.md` "Remaining" line is stale** — it lists "INT8 weight-only quant +
  wiring the OpenVINO decode session into `brain glm infer --device npu`" as
  outstanding; both are now implemented (`build_glm_int8_bytes` / `export_glm_int8`
  + `--int8`, and the `--device npu` → `npu::glm_decode::generate` route).
  Reconcile that line with the code.
- **Full-size `glm5_2()`** (78 layers, 256 experts, 154880 vocab) is **not
  runnable locally** — `config.rs` marks it for import shape validation and
  reference only; tests/training use `tiny()` and small presets.
- **Migrating GLM's dense MoE path onto `model::moe`'s sparse dispatch:
  measured, NOT done — do not migrate as-is.** `model::moe`'s new backward
  surface (this session's addition, see `docs/models/omni/status.md`) makes
  a real migration *possible* for the first time, but `crates/glm/src/
  model.rs:637-641`'s own measured naive-vs-tiled gap (1.5x-34x, worse at
  larger `m`) meant this needed measuring before touching code, not assuming
  from FLOP-counting. Measured for real on a P40 (`BRAIN_DEVICE=vulkan
  cargo run --release -p brain-glm --example moe_migration_bench`) at GLM-5.2's
  actual shape (`d_model=6144, moe_ff=2048, n_experts=256, top_k=8,
  seq_len=2048` → 64 rows/expert average): sparse-naive is **6.51x SLOWER**
  in aggregate across all 256 experts (120.98s vs 18.58s per MoE layer,
  synthetic weights) — the 32x FLOP-count win the sparse dispatch promises at
  this expert count is completely swamped by the naive kernel's per-FLOP
  inefficiency at `m=64`, landing near the bad end of that measured 1.5x-34x
  range rather than the good end. The dense path (`Mlp::Moe`'s `pick_gemm`
  -selected `matmul_reg3`) stays as the correct, faster choice **for GLM's
  shape specifically** — `crates/model/tests/moe_sparse_parity.rs` keeps
  using it as its numerical oracle, unaffected, since nothing about the
  dense path changed. The real fix is a TILED gated expert kernel (removing
  BOTH the FLOP waste and the naive-tiling gap at once) — genuinely new
  kernel work, not a migration; filed as its own follow-up
  (`.todo/moe-tiled-gated-kernel.md`), not attempted here. Until that lands,
  any future migration attempt for a DIFFERENT MoE config should re-measure
  at ITS OWN shape with `moe_migration_bench` rather than reuse this result
  — the crossover depends on `m` (rows/expert), which depends on
  `n_experts`/`top_k`/batch size, all of which vary per model.

  **Update (2026-08-09): the tiled gated kernel landed, and the decision
  flips.** `model::moe::expert_fwd_compact` (gather routed rows into a dense
  sub-batch host-side, run the SAME `pick_gemm`-selected `matmul_reg3` the
  dense path already uses on the compacted batch, scatter the scaled result
  back) is the "real fix" the paragraph above deferred — see
  `crates/model/src/moe.rs`'s "row-compacted sparse expert forward" section
  and `.todo/moe-tiled-gated-kernel.md` (now resolved). Re-run at the SAME
  real shape, same P40, `moe_migration_bench` extended with a third
  `sparse-compact` arm: **sparse-compact is 7.01x FASTER than dense-tiled**
  (2.65s vs 18.58s per MoE layer, aggregate across 256 experts) and 46.26x
  faster than sparse-naive. Migrating GLM's MoE path onto `model::moe` is now
  the measured-correct choice — see this file's own migration entry below
  (or its own dated section) for whether/how far that migration has actually
  landed as code, since a flipped BENCHMARK conclusion and a completed
  MIGRATION are two different claims.

  **Update (2026-08-09): option (a) landed — inference only.** See the
  "Done" list above (`Glm::logits_all_compact`). This is deliberately the
  MINIMAL-RISK half of `.todo/glm-model-rs-compact-moe-wiring.md`'s two
  options: a parallel forward-only function used by `sample::generate`,
  touching ZERO training code. `build_forward`/`build_backward`/
  `gradcheck::check_glm` are byte-for-byte unchanged — training still runs
  the dense path, still fully gradient-checked, nothing above regressed.
  **Option (b) (migrating `build_forward`/`build_backward` themselves, so
  TRAINING also gets the 7.01x win) is still not attempted** — it needs a
  correctly-designed row-compacted BACKWARD (dx-scatter-add semantics,
  explicitly flagged in the original `.todo` as the real design work, not
  just a mechanical port) before it can land, and should not be attempted
  without `gradcheck::check_glm` green before AND after plus its own
  mutation-verify pass, per that file's own risk framing.

## See also

- `docs/models/glm/readme.md` — architecture + status table (source of the
  `max_abs ≈ 0.005` NPU figure).
- `docs/models/glm/npu.md` — dense-expert ONNX design (its "Remaining" line is
  stale — see above).
- `docs/serving-contract.md` — the five obligations (no GLM entry yet).
- `AGENTS.md` → Models → GLM-5.2 decoder.

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

## See also

- `docs/models/glm/readme.md` — architecture + status table (source of the
  `max_abs ≈ 0.005` NPU figure).
- `docs/models/glm/npu.md` — dense-expert ONNX design (its "Remaining" line is
  stale — see above).
- `docs/serving-contract.md` — the five obligations (no GLM entry yet).
- `AGENTS.md` → Models → GLM-5.2 decoder.

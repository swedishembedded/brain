# chronos2 - roadmap

Scope: Chronos-2 time-series forecaster (`crates/chronos2`).

From-scratch reimplementation of Amazon's Chronos-2 (encoder-only T5-style
patch transformer) plugged into the forecasting API with native quantile output.
Forward parity is verified against the reference implementation.

## Done

- [x] Training: `src/train.rs` is a host (CPU) differentiable twin of the whole
      model (input embedding and `[REG]` row, every encoder block, final norm,
      quantile head), each piece finite-difference gradchecked on a tiny config.
      It is not an SSA `build_backward`; the device forward stays inference-only.
- [x] Fine-tune entry point: `src/finetune.rs` trains every weight over a
      universe of series on the shared `forecast::host_train` loop (AdamW,
      clip, batch mean) and returns the `forecast::finetune` gate verdict.
      Univariate close, horizon a whole number of output patches. Checkpoint
      read and write with the licence and base recorded.

## Not yet done

- [ ] `brain forecast finetune --chronos2`: the verb selects timesfm3 and kronos
      only; the crate API is not yet reachable from the CLI
- [ ] LoRA for the fine-tune (the host trainer has no adapter; `NoAdapter` makes
      the setting unrepresentable) and a device backward for the unscaled
      attention kernels, since host training is slow at the 120M scale
- [ ] Register `gradcheck::check_chronos2` in the fcbench harness/server
- [ ] Real covariates: user `group_ids`, past/known-future masking,
      `group_time_mask` construction, categorical target-encoding
- [ ] Long-horizon (>1024) pipeline-level autoregressive quantile-path
      unrolling
- [ ] NPU export (separate ONNX + NPU path)
- [ ] GPU kernel for multivariate group attention (currently host-computed,
      so an N-series forecast is slow)
- [ ] Known-future covariate *values* (future patches carrying covariate
      data) — only the past/multivariate path is consumed today
- [ ] Register the coalesced `rmsnorm_rows` via `block::rms_variant` instead
      of dispatching the naive per-element norm by hardcoded index (lessons
      §76 - named alongside fincast/kronos/12 other crates; measured 8.7x-
      23.5x left on the table for norms alone elsewhere in the tree)

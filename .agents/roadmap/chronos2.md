# chronos2 — roadmap

From-scratch reimplementation of Amazon's Chronos-2 (encoder-only T5-style
patch transformer) plugged into the forecasting API with native quantile output.
Forward parity is verified against the reference implementation.

## Not yet done

- [ ] `build_backward` + `impl model::Model` for training/gradcheck (today's
      forward path is inference-only, per-op-submit; training needs SSA
      buffers and a backward per op)
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

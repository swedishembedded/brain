# timesfm3 - roadmap

Google's TimesFM-3: a stacked mixing transformer with BOTH sequence attention
(over time, causal, RoPE + QK-norm) and cross-variate attention (over
variates, non-causal), CPM iterative RevIN, linear detrending and forecast
stitching. Natively multivariate - target, past-only-covariate and
known-future-covariate variates all attend to each other in one forward pass,
which is what the `ForecastModel` adapter exposes over brain's generic
`Panel`/`Role` API. Imported 1:1 from `google/timesfm-3.0-pytorch`'s
`state_dict`; forward parity (cosine ≈ 1.0, rel_l2 ≈ 3e-6) is verified layer
by layer against the real 330M-parameter checkpoint, not just end to end.
Served over the CLI, D-Bus and the resident scheduler; `rmsnorm_rows` and
`gemm_variant` are selected from the first commit rather than joining the
naive-kernel list lessons §76 keeps finding elsewhere in the tree.

The 3.0 pretrained weights carry a `timesfm-non-commercial-license-v1.0`
restriction (non-commercial, non-production, checkpoint never
redistributable) - see `docs/models/timesfm3.md`.

## Not yet done

- [ ] Training/LoRA fine-tune - no `build_backward`, no `impl model::Model`,
      no `gradcheck::check_timesfm3[_lora]` wired into
      `crates/gradcheck/tests/`. Today's path is inference-only.
- [ ] NPU export (`timesfm3_topology.rs`/`timesfm3_export.rs` +
      `npu_cli.rs`) - no Intel NPU exists on the machine this was ported on,
      so this was never started, not merely unvalidated.
- [ ] Optimization pass - no profiling has been done yet; the running-stat
      scan and CPM iterative refinement are inherently sequential over
      O(context/patch_len) patches and are the likely non-GEMM bottleneck at
      the full 15360-context / 32-variate limit.
- [ ] `crate::supply::ensure_env_weights("timesfm3")` is not called from
      `forecast_cli.rs`'s `predict`/`compare`/`serve` the way it is for
      Kronos in `predict` - `brain pull google/timesfm-3.0-pytorch` fetches
      the checkpoint into the local store, but nothing populates
      `BRAIN_TIMESFM3` from that pull automatically yet; it must be pointed
      at the fetched directory (or an imported `.safetensors`) by hand.
- [ ] Forecaster-level postprocessing covers quantile sorting and the
      positivity clamp; symmetric averaging (`(f(x) - f(-x)[::-1]) / 2`),
      full z-normalization bookkeeping and 32-variate chunking (needed to
      match the reference's own published benchmark numbers exactly on
      panels wider than the model's 32-variate limit) are not implemented.
- [ ] No per-step missing-value (NaN) interpolation - every context value is
      treated as observed.
- [ ] Left-padding is not implemented - the context length must already be a
      multiple of `input_patch_len` (32 for the published checkpoint); a
      served or CLI request with an arbitrary-length series is truncated to
      its most recent patch-aligned tail rather than left-padded.
- [ ] The D-Bus/CLI `predict` wire carries one series only, so served
      forecasting is always target-only even though the model is natively
      multivariate - full covariate support needs the library API
      (`Timesfm3Forecaster::forecast` over a `Panel`), demonstrated in
      `crates/timesfm3/examples/cooling_loop.rs`.

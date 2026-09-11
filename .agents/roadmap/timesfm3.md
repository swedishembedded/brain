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
- [x] Left-padding and per-step missing values (`Variate::observed`, or a
      non-finite value in `Variate::data`) - `preprocess::build_input` masks
      and zeroes any non-finite step, excluding it from RevIN/detrend stats;
      `Timesfm3Forecaster::forecast` left-pads any context to the checkpoint's
      `input_patch_len` boundary the same way, so the CLI no longer rounds a
      context down and silently drops history (`forecast_cli.rs`'s `predict`).
      Interior gaps are masked through, not interpolated - no attempt to
      reproduce the reference wrapper's own interior-NaN interpolation, if it
      has one; that remains a ledgered gap if a future comparison needs it.
- [ ] Numerical parity's always-on WIRING is in place (`brain-timesfm3:parity`
      in the `Makefile`'s `PARITY_STRICT_SUITES`, a `golden_tree "timesfm3"`
      entry in `scripts/data/fetch-testdata.sh`) but is UNVERIFIED: it was
      added on a box with neither `BRAIN_TIMESFM3_REF` (a
      `google-research/timesfm` checkout) nor the real checkpoint, so
      `make parity/strict` has never actually run this suite's real
      comparisons end to end, only been confirmed to hard-fail correctly when
      the fixture is absent (`BRAIN_REQUIRE_FIXTURES=1`). The new
      `padtiny`/`padtiny_full`/`nantiny` dumper cases (left-padding/missing-
      value masking) carry the same caveat, plus one unverified assumption
      spelled out in `dump_masked`'s own docstring: that a NaN placed
      directly in `target`/`past_future_covariates` is read by `decode()` as
      a missing observation, not a separate mask kwarg. Run the dumper on a
      box with both before trusting any of this. Committing the golden
      manifest itself is deliberately NOT the fix (see `715c4112`, "never
      commit large/regenerable numeric goldens").
- [x] Batched serving and native multivariate/covariate forecasting over the
      wire. `Timesfm3Forecaster::forecast` groups a panel's items by padded
      context length and issues one `core_forward` per group (mixed variate
      counts padded with wholly-NaN rows, which `build_input`'s existing
      leading-cumprod mask rule excludes as attention keys with no new
      masking logic - verified bit-identical to N single-item calls, not just
      close); `Timesfm3Instance::run_batch` groups queued D-Bus/JSONL
      invocations by horizon the same way, and its `instance_key` no longer
      forces one weight copy per horizon. `item_from_invocation` (`resident_
      forecast.rs`) reads `context` as `[T]` or `[num_target,T]` plus optional
      `past_covariates`/`known_future`/`observed`, so the model's headline
      multivariate capability is reachable over `Run`, not just the library
      API - output is `quantiles_hq` for one target (byte-identical to the
      original wire contract) or `quantiles_thq` (`[num_target,horizon,9]` +
      a `names` array) for more than one. `fcbench::score::score_windows` is
      an opt-in batched sibling of `score_split`, not yet wired into
      `backtest::run`. NOT done: a measured throughput number - this box has
      neither the real checkpoint nor representative hardware isolation
      (other processes were observed contending for the same GPU during this
      work), and this repo's own `check-no-perf-numbers.sh` gate exists
      specifically to keep an unmeasured number out of committed docs/
      comments, so none is claimed here. Run `brain perf run sweep --target
      timesfm3:<real weights> --ladder 1,2,4,8,16` (sweeping
      `BRAIN_SCHED_MAX_BATCH`) on real hardware to get one.
- [ ] Symmetric averaging, z-normalization, configurable `make_positive`, and
      32-variate chunking (the reference evaluator's own knobs) still have no
      home - `forecast::ForecastSpec` carries no per-request flags for them,
      and `Timesfm3Forecaster` has no builder-side equivalents yet either.

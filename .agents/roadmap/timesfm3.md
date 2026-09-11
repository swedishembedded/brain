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

## Done

- [x] Training/LoRA fine-tune - DONE. `crates/timesfm3/src/train.rs`
      (graph), `finetune.rs` (objective + gate), `crates/gradcheck/src/
      timesfm3.rs` (the gate), `brain forecast finetune --timesfm3` (the
      verb).
      Scope is `core_forward` only (resblock through the raw output-head
      logits; one forecast patch, no stitching/CPM feedback - see the module
      docs for why that is a real boundary, not a shortcut).

      The graph: an SSA forward bitwise-locked against `core_forward`, and a
      full hand-written backward (both attention sublayers including the
      PerDimScale fold's closed-form split, RoPE, QK-norm, FF, the resblock,
      the output head). `set_input` re-points it at the next batch in place
      (the input plus BOTH additive key-mask layouts), and `adamw_step` is
      `optim::Optim` over the trainable set - without those two the trainer
      could run one batch and never step.

      The gate is `gradcheck::check_timesfm3{,_one_layer,_lora}` plus the two
      per-ENTRY fold checks and two measured eps sweeps, wired into
      `crates/gradcheck/tests/timesfm3_fd.rs`; all six run on the default
      (Vulkan) backend AND on `BRAIN_DEVICE=cpu`. The sweep covers all 71
      tensors of the tiny config at `v != n` (variate attention is the
      sequence graph with a `[b,v,n,D] <-> [b,n,v,D]` swap around it, and at
      `v == n` a transposed adjoint stays shape-compatible with its own
      transpose, so a swapped index produces a wrong-but-well-formed buffer).
      `5e-4` is measured, not chosen: the directional U bottoms out there
      (2.21e-2 on the P40, 2.74e-2 on backend-cpu) with truncation above 1e-3
      and fp32 cancellation below 2e-4, and `timesfm3_eps_plateau` gates that
      table so a future change that moves the knee fails and prints it
      instead of inviting a widened bound.

      Six real bugs were caught here, five by a gate written before the code
      it checks:

      1. `attn_bwd_dq_bidir`/`dk_bidir` hardcode the conventional
         `1/sqrt(head_dim)` attention scale internally (no scale param),
         which is wrong for a model that always folds attention scale to 1.0
         - fixed by using `dq_bias`/`dk_bias` (which take `scale`
         explicitly) for both attention kinds, matching what
         `t5encoder::train` already does for the identical reason.
      2. `region_copy` was misused to extract a fused `d_qkv` buffer's q/k/v
         regions into dense buffers, computing destination indices against
         the FUSED buffer's stride instead of the dense buffer's own.
         `region_copy` is for same-shape/same-stride sub-region copies only;
         `unpack_qkv` (the real, purpose-built inverse of `pack_qkv`) is the
         fix.
      3. Both the forward and the training tape registered the fixed-epsilon
         `rmsnorm` kernel as `block::rms_variant`'s REFERENCE index while
         pairing it with the cooperative `rmsnorm_rows`. Those two are not
         interchangeable - `rmsnorm.wgsl` declares `Params { d_model,
         seq_len }` and hardcodes a 1e-6 epsilon, `rmsnorm_rows.wgsl`
         declares `Params { d, rows, eps }` and reads the caller's - and
         `rms_variant` picks between them purely from
         `DeviceCaps::workgroup_reductions`. So the GPU backend normalized at
         this model's own `f32::EPSILON` while the CPU backend silently
         normalized every norm in the graph at 1e-6, an epsilon roughly an
         order of magnitude larger, on a model whose QK-norm rows are
         `head_dim` wide. Pure inference `core_forward` therefore disagreed
         between the two backends by 1.76e-2 against a 1.76e-1 output scale
         on a 3-layer tiny model, and the FD gradcheck's divergence grew with
         layer count because the error compounds per norm. The fix is to
         register `rmsnorm_eps` (the same per-element reference with the
         epsilon as a parameter) instead of `rmsnorm` in both
         `model::PIPELINES` and `train::TRAIN_PIPELINES`; the GPU path is
         bit-unchanged, the CPU path now matches it to 4.5e-8.
         `timesfm3/tests/kernels.rs`'s
         `both_registered_rmsnorm_kernels_honour_the_configured_epsilon`
         gates BOTH registered indices against a host reference at this
         model's own epsilon, so the pairing cannot silently regress.
      4. The effective query gain (`query_ln.weight * log2(e) *
         softplus(per_dim_scale)`) is a function of two live parameters, not
         a tensor of its own, and it was folded ONCE at construction into a
         buffer bound into the forward's `rmsnorm_w` step - with the reverse
         folding a SECOND, independent copy. Every later write to either
         parameter was therefore invisible: an optimiser step would keep
         computing both gradients and moving both Adam moments while the
         value the graph actually used stayed pinned at its initial fold. One
         shared buffer per sublayer, refreshed by `forward()`, asserted
         bitwise against a trainer freshly built from the updated weights.
      5. The reverse writes one device temp per query-attention sublayer (the
         `rmsnorm_dw` output for the effective gain, which is not a
         ParamStore tensor and which `rmsnorm_dw` ACCUMULATES into), and
         nothing cleared it - `ParamStore::zero_grads` cannot clear a buffer
         it does not own. The second `zero_grads` + `backward` of any run saw
         exactly twice the true fold gradient, the third three times.
         Invisible to the old in-crate FD test, which rebuilt the trainer per
         probe and never ran two backwards. Cleared per BACKWARD, not per
         `zero_grads`: the host split is linear in that temp and ADDS its
         result into the parameter gradient, so carrying it across the
         microbatches of one accumulation step would fold each running total
         in again instead of each microbatch's own contribution.
      6. `checkpoint::load` PANICS on an unreadable file, so a mistyped
         `--timesfm3` path aborted the process instead of exiting 1 with a
         sentence.

      LoRA is the device-side family (`kronos::train`/`qwen3::model`'s split,
      not `model::lora`'s host `Pair`): base Frozen, `.lora_a`/`.lora_b`
      Trainable in the same ParamStore, two GEMMs and an `axpy` folding the
      delta onto each projection's OUTPUT, and the reverse recomputing
      `x.A^T` rather than caching it. Targets are the four square `[D, D]`
      projections of BOTH mixing sublayers plus `ff0`/`ff1` (ten per layer),
      with the quantile head opt-in; every one is a whole-matrix placement,
      because this architecture fuses no QKV. Matched by weight-name SUFFIX
      rather than bare leaf, since this model's leaves repeat across two
      different attention sublayers. Norm gains and `per_dim_scale` are not
      adaptable at all: a LoRA factorisation of a `[head_dim]` vector is not
      low-rank anything (for `out = 1` the product B.A has more parameters
      than the tensor it replaces), and the fold makes those two a product of
      live tensors rather than a linear map.
      `crates/timesfm3/tests/lora.rs` covers what a gradcheck cannot see - a
      fresh adapter is a BITWISE no-op, descent lowers the loss while every
      base tensor stays bit-for-bit unmoved, and folding reproduces the live
      adapted forward. That last one agrees numerically rather than bitwise,
      which is a property of the device-side family (the live path adds
      `scale*((x.A^T).B^T)` to a computed result while the folded path
      contracts against `W + scale*B.A` in one accumulation);
      `model::lora`'s host family can assert bit-equality because there
      "apply" is itself a fold into a cloned weight.

      The objective lives in `finetune.rs`, not in the trainable graph:
      `backward` takes `d_logits` from its caller because the whole output
      side (CPM-refined RevIN-reverse, a saturating clamp, stitching, trend
      re-add) holds no learnable parameters. The loss is scored on the REAL
      `preprocess::postprocess` output in original units - the number the
      served model would be judged on - and the gradient maps back through
      the same affine map: `d(loss)/d(raw) = d(loss)/d(pred) * sigma` for the
      single source patch row, exactly zero everywhere else and zero wherever
      the clamp saturated. That hand-derived Jacobian is
      finite-difference-checked against `postprocess` itself, per entry, on
      the host, including the assertion that every entry OUTSIDE the forecast
      row is exactly zero.

      `brain forecast finetune` selects its model from the flags
      (`--timesfm3` vs `--kronos-decoder`/`--kronos-tokenizer`), refuses
      both, and falls through to Kronos for neither so a bare invocation
      prints the usage block it always did. The Kronos body moved into
      `finetune_kronos` unchanged and its nine universe-loader tests pass
      unmodified. An empty VALIDATION slice is its own named failure rather
      than a "KEEP BASE" verdict: the embargoed split clears a band on both
      sides of each calendar cut, so a universe of short series can train and
      still have nothing to be judged on.
      `crates/cli/tests/forecast_finetune_timesfm3.rs` runs the real compiled
      binary against a tiny checkpoint it writes itself, so it needs no large
      fixture: 3 names, 200 bars, context 8, horizon 4, one epoch, batch 2
      gives base_val 1.1449 -> ft_val 0.5431 over 189 steps, PROMOTE.

      Licence: every artifact the fine-tune writes carries
      `timesfm-non-commercial-license-v1.0` and `variant_of:
      google/timesfm-3.0-pytorch` on its ModelCard (`variant_of` is this
      repo's existing spelling of "derived from" - it is what qwen3's adapter
      cards use for an adapter's base). A fine-tuned checkpoint is a
      DERIVATIVE of non-redistributable weights, so the terms travel in the
      file where a `cp` cannot separate them from the bytes.
      `checkpoint::license::redistributable` is the other half, consulted by
      `model_dir::register`/`register_compound` (registering a model into the
      served catalog IS redistribution - every registered model is reachable
      over the API) and by `publish_adapter`. There is no environment
      opt-out, unlike `BRAIN_FLUX2_ALLOW_NC`: that flag exists because
      FLUX.2's licence permits non-commercial USE and only the operator knows
      whether their use qualifies, which is a question about the operator;
      redistribution is not. The CLI prints a one-line notice before any
      work, so an operator about to spend hours producing an artifact they
      may not redistribute learns it in the first line.

      Two things were considered and deliberately NOT done. `impl
      model::Model` was rejected: that trait is LM-shaped
      (`vocab()`/`block_size()`) and only implemented by token-sequence
      models, and the repo's own precedent for a non-LM port is a bespoke
      trainer (`RrdbTrainer`, `T5Trainer`), which is what `Timesfm3Train` is.
      A horizon past one forecast patch is refused by name rather than
      supported: past it the forecast is stitched from several patches whose
      own RevIN statistics depend on the logits, which is a second backward
      and not a rescale.

## Not yet done

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

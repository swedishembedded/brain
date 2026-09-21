# lfm2 - roadmap

LFM2.5-Encoder: a bidirectional embedding encoder ported to brain's Rust+WGSL
engine, with chunked long-context inference, training, an NPU export path,
and a partial residency/serving integration. Forward parity is verified
against the reference implementation on both CPU and GPU backends, and
against the NPU export via OpenVINO.

## Not yet done

- [x] YaRN long-context RoPE scaling (`LfmConfig::rope_scaling`,
      `rope_base_yarn`/`rope_base_yarn_bwd` kernels) - a checkpoint can be
      configured for a 32768-token context and the kernel/wiring is
      gradient-checked and tested against the existing plain path. Quality
      at 4x the model's native 8192-token training extent is unvalidated
      extrapolation without continued pretraining.
- [x] `embed` action `normalize` param (default `false`, byte-identical for
      every existing caller) and `crates/lfm2/src/spec.rs`'s `Lfm2Spec` -
      LFM2 is now reachable from `brain::EmbeddingPipeline` as a third
      backend (routed by `ModelCard.family` for a local file, by resolver
      fallback for a hub id) - see `.agents/roadmap/embeddings.md`.
- [x] A seeded backward pass (`Lfm::seed_buf`/`backward_seeded`, mirroring
      `crates/decide/src/model.rs`'s `seed_buf`/`backward_seeded` - no
      `prepare_reverse`/staleness tracking needed here, since LFM2's step
      lists are built once at a fixed `(b, t)` rather than re-recorded per
      variable-span batch the way `decide::Encoder` is). `backward_steps`
      (the checkpoint's own MLM path) was refactored into a shared
      `trunk_backward_steps` (final RMSNorm through the tied embedding
      table) plus a small CE-only prefix; `seeded_backward_steps` is that
      SAME trunk with no CE and no head at all - an external objective's
      gradient on `xn_final` re-enters the encoder exactly where the CE
      path's own gradient used to. Gradient-checked directly against finite
      differences (`crates/gradcheck/src/lfm2_seeded.rs`,
      `lfm_seeded_analytic_grads_match_finite_differences`) over every
      parameter, isolated with the same fixed-random-linear-readout
      objective `crates/decide`'s own probe uses.
      **What this does NOT yet include**: an SDK-level training loop or
      sample that actually drives full-encoder fine-tuning with this primitive
      - `brain::EmbeddingTrainer` (`crates/sdk/src/embed_train.rs`) still only
      trains a frozen-embedding projection head; wiring it (or a sibling
      type) to ALSO unfreeze and step a live `Lfm::new_train` through
      `seed_buf`/`backward_seeded` is real, separate, not-yet-built work.
      Samples may only depend on `brain` (not `brain-lfm2` directly), so a
      sample demonstrating this needs that SDK-level surface first.
- [ ] 8k-context training: the masked-row gather before the MLM head needs a
      chunked-regime builder, since materializing full-vocabulary logits at
      8k context exceeds the device's per-buffer size limit
- [ ] Residency: length-bucketed batched `run_batch`, registration in the
      generic D-Bus executor, staged tokenize/encode/head pipelining so
      requests overlap, an NPU device lane, and batched padding (zeroed pad
      states plus an additive key mask)
- [x] Python D-Bus embedding client example - `samples/python/embedding/lfm2-embed/lfm2_embed.py`,
      exercising both `embed` (per-token hidden states + mean-pooled embedding)
      and `fill_mask` over D-Bus
- [ ] `brain perf` integration: a resident-backed concurrency benchmarking
      target, an NPU perf target, and a full model x device x concurrency
      table
- [ ] Ragged/mixed-length batched inference - only exact-length builds are
      supported today
- [ ] Registering the bidirectional/MLM encoder in the (currently
      causal-only) generic benchmarking harness
- [ ] Fixing the buffer-offset alignment violation in the GEMM attention
      fallback path at its source - it's currently just avoided because the
      faster flash-attention kernel is selected by default; a device that
      falls back to the GEMM path can still hit it

Bidirectional attention has no causal mask to hide padded positions, so
mixing sequence lengths in one batch requires either exact-length builds or
explicit zeroed pad states plus an additive mask - naive padding silently
corrupts attention scores rather than failing loudly.

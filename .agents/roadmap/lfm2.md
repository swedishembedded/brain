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
      gradient-checked and tested against the existing plain path. What
      this does NOT cover: quality at 4x the model's native 8192-token
      training extent is unvalidated extrapolation without continued
      pretraining, and it is not yet reachable from `brain::EmbeddingPipeline`
      (needs its own `crates/lfm2/src/spec.rs` `ArchSpec`, the same seam
      `qwen3::spec::Qwen3Spec` gave the Qwen3 backbone) or from an `embed`
      action `normalize` param (today's `embed` still returns the raw,
      unnormalized mean - see `crates/lfm2/src/caps.rs`).
- [ ] A seeded backward pass (`prepare_reverse`/`seed_buf`/`backward_seeded`,
      mirroring `crates/decide/src/model.rs`) for full-encoder contrastive
      fine-tuning - today only the checkpoint's own MLM objective has a
      backward path; training the encoder against an external (e.g.
      InfoNCE) objective needs a way to seed the reverse pass from outside
      the model, which does not exist yet.
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

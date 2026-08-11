# lfm — roadmap

LFM2.5-Encoder: a bidirectional embedding encoder ported to brain's Rust+WGSL
engine, with chunked long-context inference, training, an NPU export path,
and a partial residency/serving integration. Forward parity is verified
against the reference implementation on both CPU and GPU backends, and
against the NPU export via OpenVINO.

## Not yet done

- [ ] 8k-context training: the masked-row gather before the MLM head needs a
      chunked-regime builder, since materializing full-vocabulary logits at
      8k context exceeds the device's per-buffer size limit
- [ ] Residency: length-bucketed batched `run_batch`, registration in the
      generic D-Bus executor, staged tokenize/encode/head pipelining so
      requests overlap, an NPU device lane, and batched padding (zeroed pad
      states plus an additive key mask)
- [ ] Python D-Bus embedding client example
- [ ] `brain perf` integration: a resident-backed concurrency benchmarking
      target, an NPU perf target, and a full model x device x concurrency
      table
- [ ] Ragged/mixed-length batched inference — only exact-length builds are
      supported today
- [ ] Registering the bidirectional/MLM encoder in the (currently
      causal-only) generic benchmarking harness
- [ ] Fixing the buffer-offset alignment violation in the GEMM attention
      fallback path at its source — it's currently just avoided because the
      faster flash-attention kernel is selected by default; a device that
      falls back to the GEMM path can still hit it

Bidirectional attention has no causal mask to hide padded positions, so
mixing sequence lengths in one batch requires either exact-length builds or
explicit zeroed pad states plus an additive mask — naive padding silently
corrupts attention scores rather than failing loudly.

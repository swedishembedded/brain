# fincast — roadmap

FinCast: a TimesFM-style patched decoder with a sparse top-2 MoE and a
probabilistic-quantile head, imported 1:1 from the reference checkpoint.
Training, forward parity, forecaster/CLI wiring, and NPU export are all in
place; forward parity is verified against the reference implementation.

## Not yet done

- [ ] Tiled/register-blocked GEMM fast path — currently a naive `matmul` for
      CPU+GPU portability, since the tiled kernel needs workgroup barriers the
      CPU JIT backend rejects
- [ ] Per-stage parity goldens (only end-to-end parity is committed today)
- [ ] INT8/INT4 NPU QDQ calibration (the export path supports the modes; not
      yet calibrated)
- [ ] LoRA fine-tune entry point (the underlying host trainer + gradcheck
      exist, reused from Kronos, but are not wired up for FinCast)
- [ ] Register the coalesced `rmsnorm_rows` via `block::rms_variant` instead
      of dispatching the naive per-element norm by hardcoded index (lessons
      §76 - named alongside chronos2/kronos/12 other crates; measured 8.7x-
      23.5x left on the table for norms alone elsewhere in the tree)

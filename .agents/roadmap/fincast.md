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

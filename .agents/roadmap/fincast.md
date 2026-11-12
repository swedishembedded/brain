# fincast - roadmap

Scope: FinCast forecaster (`crates/fincast`).

FinCast: a TimesFM-style patched decoder with a sparse top-2 MoE and a
probabilistic-quantile head, imported 1:1 from the reference checkpoint.
Training, forward parity, forecaster/CLI wiring, and NPU export are all in
place; forward parity is verified against the reference implementation.
`src/finetune.rs` is the gated fine-tune entry point (full fine-tune of every
weight, one full context and one forecast patch per window) on the shared
`forecast::host_train` loop and the `forecast::finetune` verdict.

## Not yet done

- [ ] Tiled/register-blocked GEMM fast path — currently a naive `matmul` for
      CPU+GPU portability, since the tiled kernel needs workgroup barriers the
      CPU JIT backend rejects
- [ ] Per-stage parity goldens (only end-to-end parity is committed today)
- [ ] INT8/INT4 NPU QDQ calibration (the export path supports the modes; not
      yet calibrated)
- [ ] `brain forecast finetune --fincast`: the verb selects timesfm3 and kronos
      only; the crate API is not yet reachable from the CLI
- [ ] LoRA for the fine-tune (the host trainer has no adapter; `NoAdapter` makes
      the setting unrepresentable), a padded-context and multi-patch horizon
      fine-tune, and a device backward, since host training is slow at the
      50-layer scale
- [ ] Register the coalesced `rmsnorm_rows` via `block::rms_variant` instead
      of dispatching the naive per-element norm by hardcoded index (lessons
      §76 - named alongside chronos2/kronos/12 other crates; measured 8.7x-
      23.5x left on the table for norms alone elsewhere in the tree)

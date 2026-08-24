# codeformer - roadmap

CodeFormer blind face restoration (`crates/codeformer`): builds on `crates/vqgan`'s
VQ autoencoder and adds the code-prediction transformer, the controllable
feature transformation (CFT), and the fidelity dial `w`. Forward parity is
verified against the reference implementation, per stage and across the `w`
sweep, and the serving contract (`codeformer::caps` `restore_face`,
`resident_restore::RestoreResident`, a `catalog.rs` entry, D-Bus `Run`,
`examples/restore/restore_face.py`) is in place.

Backward is done and gated: `gradcheck::check_codeformer` (the
code-prediction Transformer under the CFT, with the VQ autoencoder frozen -
its own backward is `check_vqgan`) and `check_codeformer_one_layer`, wired in
`crates/gradcheck/tests/imaging_models.rs`.

## Not yet done

- [ ] `adain=True` support (the upstream reference's own default inference
      path normalizes the predicted codes toward the input's statistics
      before the generator; only the `adain=False` path is implemented)
- [ ] Face detection / alignment wired into the pipeline - the existing
      face-alignment code targets a different landmark template than this
      model expects, so it can't be reused as-is; the restorer currently
      takes an already-aligned face
- [ ] Batch > 1 support in the forward graph. Batch size is hardcoded to 1 in
      the shared `vae::blocks` builder this crate and the diffusion VAE both
      use, so the fix lands **there**, not as a codeformer-only fork - and
      pays off for every VAE-family model at once
- [ ] Input sizes other than 512x512
- [ ] Performance profiling / optimization pass

Not gaps, recorded so they stop being re-opened:

- Background upsampling / paste-back is a separate super-resolution model,
  and it now exists: `crates/rrdbnet` (Real-ESRGAN) is wired into `imgpipe`
  as the pipeline's `UPSCALE_MODEL` stage.
- There is deliberately no `brain codeformer` CLI module - the model is
  reached generically as `brain codeformer restore_face`, which is what the
  serving contract asks for.
- `run_batch` is the serial default *with a stated reason*
  (`resident_restore.rs`: batching is no better a trade than running twice),
  which the contract permits.

The position embedding is a fixed-size parameter with no interpolation, so
the architecture itself is pinned to one input resolution rather than being
resizable at runtime.

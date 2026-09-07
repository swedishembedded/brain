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
  which the contract permits. This does NOT mean the forward graph itself
  cannot batch (see below) - it means the serving contract chose not to wire
  that capability up, for a reason unrelated to whether it exists.

The position embedding is a fixed-size parameter with no interpolation, so
the architecture itself is pinned to one input resolution rather than being
resizable at runtime.

**Batch > 1 in the forward graph is done.** The earlier note here ("batch
size is hardcoded to 1 in the shared `vae::blocks` builder") was imprecise:
every batch-sensitive kernel the builder dispatches has ALWAYS taken an
`N`/`bsz` parameter and indexed per-sample correctly - the gap was purely
that `vae::blocks::Builder` hardcoded that parameter to `1` at every dispatch
site. That is fixed (`Builder::set_batch`), and it is what `CodeFormer::
new_batched` builds on for the encoder and generator ([`vqgan::model::
run_blocks`], shared with `crates/vqgan`). The code-prediction Transformer
and the controllable feature transformation are NOT `vae::blocks` - they
dispatch `attn_scores_bidir`/`_softmax`/`_apply` and the CFT's elementwise
kernels directly - so `new_batched` threads `n` through those by hand too:
the Transformer's self-attention gets a real `bsz` Params field (exactly what
crossing batch elements there would otherwise risk, the same failure mode
`vae::blocks::Builder::attn`'s own self-attention had to avoid), and its
position embedding - one `[T,E]` tensor shared by every image, not itself
batched - is broadcast-added per image rather than tiled into a host buffer.

The one exception is `add_chan` (`sdxlunet`'s per-image timestep-embedding
broadcast), which stays pinned to batch 1 for a reason specific to
`sdxlunet`'s own bias upload - irrelevant here, since this crate's graph
never records that op, but worth knowing before assuming every op the shared
builder registers is now batch-general.

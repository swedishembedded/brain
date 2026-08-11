# restore — roadmap

CodeFormer blind face restoration (`crates/restore`): builds on `crates/vqgan`'s
VQ autoencoder and adds the code-prediction transformer, the controllable
feature transformation (CFT), and the fidelity dial `w`. Forward parity is
verified against the reference implementation, per stage and across the `w`
sweep, and the basic serving contract (single-image `restore_face` over
D-Bus) is in place.

## Not yet done

- [ ] Backward pass / gradcheck for the transformer and CFT — the
      convolutional encoder/generator can reuse existing block adjoints, but
      the transformer's `qk`/`v` split needs its own backward stitch and the
      CFT block needs a couple of adjoint kernels (a `concat2` backward, and
      a reduction for the fidelity-dial gradient) that don't exist yet
- [ ] `adain=True` support (the upstream reference's own default inference
      path normalizes the predicted codes toward the input's statistics
      before the generator; only the `adain=False` path is implemented)
- [ ] Face detection / alignment wired into the pipeline — the existing
      face-alignment code targets a different landmark template than this
      model expects, so it can't be reused as-is; the restorer currently
      takes an already-aligned face
- [ ] Background upsampling / paste-back (a separate super-resolution model;
      out of scope for this crate)
- [ ] Batch > 1 support in the forward graph
- [ ] Input sizes other than 512x512
- [ ] CLI subcommand (`brain restore`)
- [ ] Batched serving (`run_batch`) — requests are served serially today
- [ ] Performance profiling / optimization pass

The position embedding is a fixed-size parameter with no interpolation, so
the architecture itself is pinned to one input resolution rather than being
resizable at runtime.

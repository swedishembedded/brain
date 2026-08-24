# pulid - roadmap

PuLID-FLUX identity conditioning (`crates/pulid`): the `IDFormer` Perceiver
resampler (face embedding → 32 ID tokens) and the injected
`PerceiverAttentionCA`, cross-attended into the FLUX.1 image stream at 20
sites. Composes `clip::EvaVision` and `flux1::{Flux, inject}`; adds no kernel
and no shared block.

Parity-gated on both backends against a hooked reference - IDFormer 29 taps,
the CA unit 8, and the conditioned FLUX.1 forward 10, worst 1−cos 1.44e-11.
The image → `id_cond` path exists (`idcond::IdCond::from_image` /
`idcond::compose`), and the serving contract is met: `pulid::caps`
(`text2image`), `resident_pulid::PulidResident`, a `catalog.rs` entry, D-Bus
`Run`, `examples/imagegen/pulid_generate.py`.

## Not yet done

- [ ] Reference-grade face preprocessing. The served path resizes the face
      crop straight to EVA-CLIP-L/336 instead of reproducing the reference's
      RetinaFace + BiSeNet alignment/parsing (`caps.rs`'s module docs). This
      is the one documented numeric divergence from upstream.
- [ ] `id_weight` / `start_step` sweep against a real identity-fidelity
      metric
- [ ] Full-depth conditioning run across all injection sites - only a
      reduced-depth run has been exercised; an int8 full-depth run is
      possible in principle but has not been done
- [ ] Multi-image identity conditioning - only a single reference embedding
      per identity is supported
- [ ] Backward pass / gradient check for the adapter (`check_pulid`)
- [ ] Batch > 1 - serial `run_batch`, same reason as `flux1`'s (one
      multi-step sample per request)

The current forward pass reuses buffers across layers in an inference-shaped
way, so a backward pass needs a distinct training-mode forward with per-layer
buffer allocation rather than a flag on the existing one. That refactor is
the prerequisite for `check_pulid`, not an afterthought to it.

Only `dev` is validated against a PuLID reference, and no reference dump of a
full ID-conditioned *generation* exists in this workspace - so "PuLID works"
end to end is not a claim this crate supports.

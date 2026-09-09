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

Ordered by expected identity-quality impact per fix, not by ease - reference-
grade preprocessing is upstream of the ENTIRE EVA-CLIP half of the identity
representation (five tapped hidden states plus the CLS embedding), so it is
listed first even though it is also the largest single item here.

- [ ] Reference-grade face preprocessing. The served path resizes the face
      crop straight to EVA-CLIP-L/336 instead of reproducing the reference's
      RetinaFace + BiSeNet alignment/parsing (`caps.rs`'s module docs). This
      is the one documented numeric divergence from upstream, and the
      highest-leverage open item: BiSeNet face parsing is effectively a new
      model port (weights, import, a parity ladder of its own) under this
      repo's own porting discipline, not a small preprocessing tweak.
- [x] `start_step` - `Flux1::generate_injected` now takes `GenerateOptions
      .start_step` (steps before it forward WITHOUT injection, identical to
      `inject: None`); `pulid::caps`'s `text2image` exposes it (default 0 =
      every step, matching the prior always-inject behavior). Still open:
      sweeping it (and `id_weight`, whose default/range now match upstream's
      own 1.0/0..3 rather than an unexplained 0.8/0..2) against a real
      identity-fidelity metric - `examples/imagegen/identity_score.sh`'s
      ArcFace-cosine method is the tool, no real run has swept it yet.
- [ ] Full-depth conditioning run across all injection sites - only a
      reduced-depth run has been exercised; an int8 full-depth run is
      possible in principle but has not been done
- [ ] Multi-image identity conditioning - only a single reference embedding
      per identity is supported (upstream's own PuLID v1.1 takes a primary
      image plus up to three auxiliary ones)
- [ ] True CFG - upstream PuLID-FLUX optionally runs a second, unconditioned
      FLUX forward (negative prompt, negative T5/CLIP conditioning, an
      unconditioned ID embedding, `timestep_to_start_cfg`) and combines
      `negative + true_cfg * (positive - negative)`, on top of the cheaper
      distilled guidance scalar this crate already has. `flux1::pipeline`
      has no CFG branch of any kind to build this on today
      (`pipeline.rs`'s own module docs). Upstream's own docs say the
      distilled guidance alone is usually enough, which is why this sits
      below full-depth/multi-image rather than above them.
- [ ] `FLUX.1-Krea-dev` as a validated PuLID variant - upstream added this
      combination; `pulid::caps::VARIANTS` lists `kontext-dev`/`schnell`
      (architectural extrapolations, explicitly NOT upstream-supported
      PuLID combinations) but not `krea-dev`.
- [ ] Backward pass / gradient check for the adapter (`check_pulid`)
- [ ] Batch > 1 - serial `run_batch`, same reason as `flux1`'s (one
      multi-step sample per request)

## Explicitly out of scope for this crate

**PuLID v1 / v1.1 on SDXL is a separate model, not a missing feature of this
one.** Upstream's SDXL branch is its own attention-processor/IDFormer
integration (community-checkpoint support, up to three auxiliary identity
images, its own CFG/`num_zero`/orthogonalization/sampler options) built on
`crates/sdxlunet`, not on `crates/flux1`. If it is ever wanted, it is a new
crate with its own import/parity ladder and its own roadmap entry under this
repo's normal porting discipline - not an extension of `crates/pulid`, which
stays FLUX.1-only.

The current forward pass reuses buffers across layers in an inference-shaped
way, so a backward pass needs a distinct training-mode forward with per-layer
buffer allocation rather than a flag on the existing one. That refactor is
the prerequisite for `check_pulid`, not an afterthought to it.

Only `dev` is validated against a PuLID reference, and no reference dump of a
full ID-conditioned *generation* exists in this workspace - so "PuLID works"
end to end is not a claim this crate supports.

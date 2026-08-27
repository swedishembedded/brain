# PuLID-FLUX

Identity-conditioned image generation on [FLUX.1](flux1.md): an `IDFormer`
Perceiver resampler (a face embedding -> 32 ID tokens) cross-attended into
the FLUX.1 image stream at 20 sites, added to the residual stream (never
concatenated as tokens). Composes [ArcFace](arcface.md)'s embedding and
EVA-CLIP; adds no new kernel.

This is a real, verified port - 312 tensors, parity-gated on both backends
(IDFormer 29 taps, the cross-attention unit 8, the conditioned FLUX.1
forward 10, worst 1-cos 1.44e-11).

It is **served**: a capability manifest (`text2image`, model id
`brain/flux1-pulid`), a residency adapter (`BRAIN_FLUX1_DIR` for the FLUX.1
backbone, `BRAIN_PULID_DIR` for the PuLID checkpoint, `BRAIN_ARCFACE_DIR` and
`BRAIN_CLIP_DIR` for the identity towers), D-Bus `Run`, and a runnable
example (`examples/imagegen/pulid_generate.py`) - a prompt plus a face photo
in, an identity-conditioned image out. The image -> `id_cond` path is wired
at this serving layer (an ArcFace embedding plus EVA-CLIP taps composed by
`crate::idcond::compose`), a documented approximation of the reference
preprocessing rather than the real thing: no RetinaFace alignment or BiSeNet
face parse exists in this workspace, so the same face crop ArcFace uses is
resized straight into EVA-CLIP with no parsing (see
`crates/pulid/src/caps.rs`'s module docs for the gap).

No backward/training path exists, and end-to-end verification is doubly
unconfirmed: it composes [FLUX.1](flux1.md)'s own not-yet-verified
sampler-loop glue with PuLID's own unverified injection wiring, so there is
no reference dump of a full PuLID-conditioned generation to check against
yet. "PuLID works" is not claimed.

Package: `brain-pulid`.

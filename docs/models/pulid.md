# PuLID-FLUX (not yet servable)

Identity-conditioned image generation on [FLUX.1](flux1.md): an `IDFormer`
Perceiver resampler (a face embedding -> 32 ID tokens) cross-attended into
the FLUX.1 image stream at 20 sites, added to the residual stream (never
concatenated as tokens). Composes [ArcFace](arcface.md)'s embedding and
EVA-CLIP; adds no new kernel.

This is a real, verified port - 312 tensors, parity-gated on both backends
(IDFormer 29 taps, the cross-attention unit 8, the conditioned FLUX.1
forward 10, worst 1-cos 1.44e-11) - but forward only: no backward, no
serving surface, and no image -> `id_cond` path exists yet (the crate
takes `id_cond` as raw host slices, so the arcface/EVA-CLIP wiring isn't
done). "PuLID works" is not claimed. Not something you can run as a model
today.

Package: `brain-pulid`.

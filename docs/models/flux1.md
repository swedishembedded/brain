# FLUX.1 / Kontext (not yet servable)

Black Forest Labs' 12B MMDiT: 19 double-stream blocks (separate img/txt
weights, joint attention) then 38 single-stream parallel blocks, per-block
modulation, T5-XXL + CLIP-L conditioning, plus the Kontext reference-image
editing path. [PuLID](pulid.md) conditions on this backbone for identity
injection.

This is a real, verified port - imported (1160 -> 780 tensors, two-way
covered), forward-parity-gated (reduced-depth fp32 worst 1-cos 1.5e-11;
full-depth int8 out cosine 0.9985/0.9991) - but transformer-forward only:
no sampler loop, no VAE glue, no CLI, no serving surface, and no backward/
gradcheck. Not something you can run as a model today.

Package: `brain-flux1`.

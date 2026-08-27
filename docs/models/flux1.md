# FLUX.1 / Kontext

Black Forest Labs' 12B MMDiT: 19 double-stream blocks (separate img/txt
weights, joint attention) then 38 single-stream parallel blocks, per-block
modulation, T5-XXL + CLIP-L conditioning, plus the Kontext reference-image
editing path. [PuLID](pulid.md) conditions on this backbone for identity
injection.

This is a real, verified port - imported (1160 -> 780 tensors, two-way
covered), forward-parity-gated (reduced-depth fp32 worst 1-cos 1.5e-11;
full-depth int8 out cosine 0.9985/0.9991).

It is **served**: a capability manifest (`text2image`, model id
`brain/flux1`), a residency adapter (`BRAIN_FLUX1_DIR`, a released diffusers
FLUX.1 checkpoint root), D-Bus `Run`, and a runnable example
(`examples/imagegen/flux1_generate.py`), on top of a complete sampler loop
(T5-XXL + CLIP-L conditioning, FLUX.1's own linear-shift schedule, VAE
decode). The glue this loop adds on top of the parity-gated DiT/T5/CLIP/VAE
pieces - patchify layout, position ids, the schedule, the affine latent
normalization - has **no end-to-end real-weight verification** in this
workspace yet (see `crates/flux1/src/pipeline.rs`'s own note on it): treat a
first real generation as the test of that glue.

Kontext reference-image editing, img2img, LoRA adapters, and backward/
gradcheck are all deferred - this is a single-image text-to-image loop
today, with no batching or INT8 on the serving path either.

Package: `brain-flux1`.

# SDXL UNet2DConditionModel

The diffusion backbone [ControlNet](controlnet.md)'s SDXL producer
conditions: `CrossAttnDownBlock2D`/`DownBlock2D` -> mid -> up with skip
concats, spatial transformers on the two inner levels, and the SDXL added
conditioning (pooled text + time-id sinusoids).

This is a real, verified port - imported (1680 -> 1610 tensors, two-way
covered) and forward-parity-gated (165 comparisons, worst cosine
0.9999999999, `out.sample` cosine 1.0000000000).

It is **served**: a capability manifest (`text2image`), a residency adapter
(`BRAIN_SDXL_DIR`), D-Bus `Run`, and a runnable example under
`examples/imagegen/`, on top of a complete sampler loop (dual CLIP
conditioning, a discrete Euler step, CFG, VAE decode). It is also
**trainable**: the backward is gated by finite differences over the whole
graph, including a per-entry check on the timestep-embedding chain that all
17 resnets share.

Two things it still does not do: batching (every request is its own
multi-step sample, so concurrent requests are served serially) and INT8.

Package: `brain-sdxlunet`.

# SDXL UNet2DConditionModel (not yet servable)

The diffusion backbone [ControlNet](controlnet.md)'s SDXL producer
conditions: `CrossAttnDownBlock2D`/`DownBlock2D` -> mid -> up with skip
concats, spatial transformers on the two inner levels, and the SDXL added
conditioning (pooled text + time-id sinusoids).

This is a real, verified port - imported (1680 -> 1610 tensors, two-way
covered), forward-parity-gated (165 comparisons, worst cosine 0.9999999999,
`out.sample` cosine 1.0000000000) - but forward only: no capability
manifest, no residency adapter, no `run_batch`, no D-Bus, no example, no
CLI, no sampler loop, no VAE/text-encoder glue, and backward is deferred.
Not something you can run as a model today.

Package: `brain-sdxlunet`.

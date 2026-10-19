# AutoencoderKL (component)

The diffusers `AutoencoderKL` variational autoencoder - the shared latent
VAE behind [Z-Image](s3dit.md), [FLUX.2 Klein](flux2.md) and
[SDXL UNet](sdxlunet.md). Lives in `crates/vae`, which is also where the
shared conv-block `Builder` other image models
([VQGAN](vqgan.md), [RRDBNet](rrdbnet.md)) build on lives - kept as one
crate rather than split, so that shared infrastructure has exactly one
home instead of two.

Not independently servable: no capability manifest or CLI verb of its own,
reached only through the models that decode/encode through it.

## High resolution: tiling

A VAE graph's activations scale with the IMAGE, not with the checkpoint, so
the decode is where a high-resolution run dies - and it dies last, after every
denoise step has been paid for. At FLUX.2's channel schedule a 1024x1024
decode holds about 11 GiB and a 2048x2048 one about 41 GiB, which no 24 GiB
card can do.

Past a device budget both directions switch to `vae::VaeTiledDecoder` /
`vae::VaeTiledEncoder`: an overlapping cover of 512-pixel tiles, one device
graph resident at a time, blended with trapezoidal masks. Peak VRAM becomes
the tile's - about 3.4 GiB - however large the image is, and
`vae::decoder_device_bytes_for_pixels_planned` is what a placement decision
reads so the plan describes the graph that will actually be built.

That budget is what the target card has FREE right now
(`vae::tiled::whole_graph_budget`, over `gpu_core::capacity`'s live probe),
less the headroom between a byte estimate and a real allocation - not a
constant. A card's size is not what a graph may spend: an img2img generation
holds a quantised DiT of several GiB on the same card for the whole run, so a
graph that fits an empty 24 GiB card need not fit that one, and a graph that
fits neither may fit a larger card outright. `vae::tiled::WHOLE_GRAPH_MAX_BYTES`
remains as the fallback for a device nothing can measure (no card, no driver
query, the host tier).

Below the threshold nothing changes: a cover that does not split is the
whole-image path bit for bit. Above it the result is close but not identical,
and the number is measured rather than assumed - at 1024x1024 in nine tiles,
38.5 dB against the whole-image decode, which is this VAE's own reconstruction
accuracy. Getting there needs the GroupNorm statistics synchronised across
tiles (a first pass that only measures them); without that the same cover is
26.5 dB, the per-tile brightness stepping naively tiled VAEs are known for.
`BRAIN_VAE_TILE=1`/`0` forces the decision either way.

Package: `brain-vae`.

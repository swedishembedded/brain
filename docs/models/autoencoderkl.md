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

Package: `brain-vae`.

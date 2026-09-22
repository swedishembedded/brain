<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 16. A config default must mirror the REFERENCE's default, not "off"

diffusers' `AutoencoderKL.__init__` declares `use_quant_conv: bool = True,
use_post_quant_conv: bool = True`, and a `config.json` only carries the keys it
**overrides** - so the entire SDXL/SD1.x family omits both and means *true*.
`VaeConfig::from_json` defaulted them to **false**, which silently dropped
SDXL's `post_quant_conv`: a 1x1 mixing of the four latent channels, applied
before `decoder.conv_in`.

The failure is the instructive part. Dropping a channel mixing leaves the
decode **in a perfectly plausible `[-1, 1]` range** and **uncorrelated with the
truth** - cosine **-0.03** against the reference. The picture kept its
structure (a fox, a forest, snow) and had unusable colour, because structure
comes from the latent and the ruined part was a linear recombination of it.

Three things let it ship:

* No gate. `crates/vae`'s only decode parity covers *Z-Image's* 16-channel
  `AutoencoderKL` and skips unless `BRAIN_S3DIT_VAE` is set, so nothing gated
  a VAE decode of this kind at all (#1, again).
* The UNet's own 165-tap parity was green - and stayed green. It ends at the
  latent; the defect is downstream of everything it checks.
* The sanity check applied to the output was **mean gradient magnitude**, which
  "photo range" confirms for any textured image, including a wrong one. A
  statistic that a broken result also satisfies is not a check.

Read the reference's constructor signature, not just the checkpoint's JSON. Any
model that genuinely wants these off says so explicitly (FLUX.1-Kontext and
Z-Image both write `false`); the ones that say nothing are the ones that need
`true`. Gated by `crates/vae/tests/sdxl_decode_parity.rs`, now at cosine
1.000000 / PSNR 121 dB.

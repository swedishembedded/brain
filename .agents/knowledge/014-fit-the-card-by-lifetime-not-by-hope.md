<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 14. Fit the card by lifetime, not by hope

SDXL is ~3.5 B params across four models (~14 GB at fp32), and - under the
default wgpu backend - a non-ReBAR Pascal card carries roughly 2x resident
overhead per uploaded storage buffer (see #35: this turned out to be specific
to wgpu's Vulkan HAL, not the hardware - `--device vulkan` measures 1x on the
same card). Holding the UNet, both text encoders and the VAE at once does not
fit 24 GB.

The fix is scheduling, not precision: the text encoders are needed ONCE per
generation and the VAE once at the end, while the UNet runs every step. Build
the encoders, encode every prompt (conditional AND unconditional) in one pass,
drop them; run the loop; then decode.

The failure mode is worth the entry: the first version OOMed **at the VAE decode
after all 24 steps had already run** - the most expensive possible moment to
discover a memory problem. Prefer decoding on the CPU when the resident model
still owns the card; it is one pass.

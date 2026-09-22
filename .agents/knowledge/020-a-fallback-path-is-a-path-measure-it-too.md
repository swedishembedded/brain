<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 20. A fallback path is a path - measure it too

`vae::blocks::gn` picks a cooperative GroupNorm reduction when the device has
workgroup reductions and a serial one otherwise. The cooperative branch was
measured and tuned; the FALLBACK was never measured at all, and `backend-cpu`
reports `workgroup_reductions: false`, so every conv-autoencoder in the tree -
vae, vqgan, restore, unet, flux1 - ran the unmeasured branch on the CPU JIT.

It was the serial kernel: `g` = 32 invocations for up to 33 M elements. A
barrier-free two-stage reduction measured **~3x** faster at every VAE decoder
shape, and it already existed - `crates/wm-diamond` had written it privately
after measuring the serial one at 77% of its frame time.

Three things generalise:

* **Profile the branch your hardware does NOT take.** A capability-gated
  fallback is invisible on the machine that never takes it, which is exactly
  where an unmeasured path hides.
* **Faster can be more accurate.** Summing 33 M elements in one lane loses
  precision a two-stage reduction keeps: SDXL's VAE decode parity went from
  PSNR 121.39 dB to 127.95 dB at the same cosine. Speed and accuracy are not
  always a trade.
* **A/B harnesses need their own sanity check.** The first run of this one
  reported the three kernels disagreeing by 1.7 - which was the harness
  dispatching `gn_stats_wg` at 64 threads when it is `@workgroup_size(256)`.
  It was caught only because the VAE parity gate was green, so the harness had
  to be the wrong one. Compare against a HOST oracle, not just kernel-to-kernel.

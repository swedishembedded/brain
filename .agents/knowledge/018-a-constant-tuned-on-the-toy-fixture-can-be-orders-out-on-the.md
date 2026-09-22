<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 18. A constant tuned on the toy fixture can be orders out on the real model

`crates/upscale`'s tile halo is the cost/quality knob for super-resolution
tiling: too small and each tile is computed as if the image ended at its border,
which shows as a grid of seams. The first draft picked **16**, and the
checkpoint-free 2-block gate agreed - max |seam| **9.2e-4**, four times below an
8-bit quantisation step, zero visible pixels. It looked measured, because it was.

On the released 23-block `x4plus` the same halo measures **7.3e-1**: three
orders of magnitude worse, 45 676 visibly wrong pixels. The reason is
structural - a 3x3-conv net's receptive radius grows with DEPTH (~`1 + 15*blocks
+ 1`, so ~32 input pixels at 2 blocks and ~347 at 23) - so the toy could not have
predicted the real number no matter how carefully it was measured.

Two things follow, neither specific to upscaling:

* A **checkpoint-free gate is necessary and not sufficient.** It runs everywhere
  and catches wiring, shapes and algebra; it cannot calibrate anything whose
  scale depends on the real model's depth or width. Any constant of that kind
  needs a measurement on the released weights, even when that gate has to skip
  on most machines.
* **Report the sweep, not the chosen value.** The table in `TILE_HALO`'s doc
  comment shows both configs at every halo tried, so the next person can see
  that the number is a trade-off with a known cost.

And a corollary earned the hard way, in the same file: **the obvious remedy was
wrong.** "Hard-cropped tiling leaves a seam, so blend the overlap instead" is
the standard move, and it was written into the doc comment as the planned fix
before anyone measured it. Blending is *worse* - 2.1e-2 against cropping's
3.3e-6 on the toy, 2.0e-1 against 1.6e-1 on the released net - because it mixes
each tile's halo pixels, the least accurate ones it computed, back into the
output, where cropping throws them away and keeps the well-conditioned interior.
Blending trades the error's magnitude for its continuity. A planned fix recorded
in a comment is a claim like any other; this one is now recorded as refuted, with
the numbers, so it is not attempted a third time.

The comparison also had to be set up correctly to mean anything: tiled-vs-whole
image is NOT a seam measurement, because the whole-image path lets the
convolutions zero-pad at the image border while any tiled path replicate-pads
it. Holding the border regime fixed - tiled vs ONE tile covering everything -
is what isolates the seam.

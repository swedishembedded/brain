<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 15. A kernel-selection GUARD can be the bottleneck

`block::pick_gemm` fell back to the naive GEMM when `m < 128 || n < 128`, on the
reading that a partial 128x128 tile is wasted. The tiled kernel bounds-guards its
tile, so a short `M` costs only the idle rows - while the naive kernel gives one
thread per output element, each walking `k` serially, which collapses on a wide
`N`.

SDXL's cross-attention `kv` projection is `[77, 2048, 2560]`. 77 text tokens is
under the old threshold, so **60 of those per forward took the naive path at
43 GFLOP/s - 0.4% of the roof, and 49% of the entire UNet forward**. Measured
crossover (k=2048, n=2560): naive wins to m=4, tiled from m=8, and by m=77 tiled
is **22x** (18.67 ms -> 0.84 ms), bit-identical.

Lowering the guard to `m < 8` took the whole UNet forward from **2609 ms to
1349 ms (1.93x)** and the end-to-end 24-step generation from 174 s to 105.7 s,
with the output **bit-identical** at the same seed.

Two general points:

* This is the third instance of #8 in one workstream - a fast
  kernel that existed and was not reached. Twice it was an unregistered sibling;
  here it was a *selection rule* that excluded it. Profile the selection, not
  just the kernels.
* At `m = 1` neither is right: naive beat tiled 0.19 ms vs 0.48 ms, and the
  correct kernel is `matmul_gemv` (one workgroup per output column), which
  `gemm_variant` selects for models that register it. `crates/unet` does not, so
  its conditioning path still runs naive - small (~6 ms) but real.

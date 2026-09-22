<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 27. A "% of peak" divided by a hardcoded peak is not a measurement

Every profiler in this tree reported utilisation against a literal:
`PEAK_TFLOPS`/`PEAK_GBPS`/`PEAK_FP32` constants hardcoded in `vqgan_bench`,
`unet_bench`, `zimage_bench` and several microbenches - one card's spec-sheet
numbers. Three separate problems follow, and the third is the one that matters:

1. **On any other device every number is wrong**, silently and by an unbounded
   factor. `DeviceCaps::peak_bandwidth_gbs` existed for exactly this and was
   `None` on all three backends, with nothing anywhere filling it.
2. **Spec-sheet peak is not achievable peak.** Measured with a
   dependency-free FMA probe and a STREAM-triad probe, achieved throughput came
   in at roughly 85-90% of the spec-sheet number. Grading kernels against a
   roof nothing can reach builds a permanent 10-17% pessimism into every row.
3. **The whole method depends on the denominator.** `.agents/rules/kernels.md`
   §F is "rank against the roof, fix the top row, re-profile". A wrong roof does
   not produce an obviously wrong answer - it produces a plausible one, and
   quietly invalidates the ranking that everything else is built on.

`gpu_core::roof` measures both roofs once per adapter (persisted with
`gpu_core::tune`'s key discipline, so editing a probe invalidates old numbers by
construction) and `Gpu::caps()` overlays them. The probe measures the
**silicon**, deliberately not "the best GEMM we have written" - a roof derived
from `matmul_reg3` would hide precisely the gap the workstream exists to close.

Corollary worth stating separately: **do not measure a roofline under
contention.** Two probes sharing a device measure the contended device and
disagree by more than 25%, which broke a reproducibility test under heavy
parallel test execution and is not a bug in the probe.

<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 410. Two consumer warpgroups sharing a TMA ring faulted only under back-to-back launches, and the sanitizers did not see it

Symptom: a served run (16K x 64, brain's defaults) died with `cuStreamSynchronize failed: unspecified launch failure`; the
kernel log said Xid 13, `Graphics SM Global Exception ... Multiple Warp Errors`, then Xid 43 (the channel torn down). The same
fault had hit a server's first tuning pass and a sweep of `native_check_fp8_shape` the same day; ECC counters were clean.

How it was found (each step measured, not reasoned):
- compute-sanitizer memcheck over the failing sweep: 0 errors; synccheck: 0; racecheck flagged RAW hazards on the combine buffer
  of `matmul_fp8_wgmma_swap_tma_bn80` whenever a block ran more than one tile (it may not model the `bar.arrive`/`bar.sync`
  hand-off the code uses), but the sanitized runs never faulted: the fault needs the timing of uninstrumented launches.
- `%smid` aliasing (the map slots are keyed by `smid % 132`) was ruled out on the card: smid runs 0..131, nsmid 132.
- A stress loop of back-to-back timings of one candidate at a time (`native_time_fp8_candidates`, rows 1-64, the 5120 x 34816
  and 17408 x 16384 products) reproduced it: `swap_tma_bn136` (two consumer warpgroups) and `swap_tma_bn96` (two) faulted in
  1-2 of 4 runs each, at 17-64 rows as well as 1-16 (so not caused by offering the kernel from one row); the one-consumer
  `bn136c1` and `bn144c1` ran 4 x 120 iterations and the 5-minute reproducer without a fault; the prefill `tma_n256` survived
  3 x 120 randomized iterations.

Action: the three two-consumer candidates (bn80, bn96, bn136) and the kernel's two-consumer path are removed; the one-consumer
forms were within ~1 % of them at the served shapes (64-row gate/up 63.1 vs 62.4 us). Reproducer:
`swap_tma_back_to_back_stress` (ignored test in `crates/gpu-core/tests/it/fp8_linear_native.rs`): a new swap candidate must survive
it before it joins. Open: the root cause in the two-consumer design (alternate K blocks on an odd-depth ring, the combine
hand-off, `setmaxnreg` 40/232) is not identified.

Rule: a candidate is admitted on agreement with the reference AND on surviving back-to-back launches of itself at the served
shapes; the admission check alone (one launch per shape) and the sanitizers (instrumented timing) both missed this fault.

<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 399. A TMA-fed decode GEMM reaches 2.9 TB/s on the wide products, and the 1.3 PFLOPS the reference server seems to run at 8192 rows is an average over small rounds, not a rate

Question (knowledge 387, 393, 398): the FP8 GEMM is 54 % of a 31K prefill and 55 % of a c=64 mixed iteration; the reference server's profile says gate/up
8192 x 34816 x 5120 takes 2.095 ms (1.39 PFLOPS) and the 64-row gate/up 58.9 us (3.0 TB/s) against brain's 1.12 PFLOPS and 73.9 us. Which of those
are rates the card gives, and what moves brain's?

## What the reference actually runs (same card, same session)

- The 2.095 ms is the **mean over 2,847 launches** (min 0.197, median 2.409, max 2.664 ms): rounds of every size. At 8192 rows the same kernel
  (the reference's `sm90_fp8_gemm_1d2d`, tile 256 x 128, 2-block multicast of the weight tile, bf16 output) measures, isolated on gpu1 with weights cycled
  past the L2 and a Gaussian activation: gate/up 1165 TFLOPS, qkv 1165, down 1299, out 1292; at 2048 rows gate/up 1270, qkv 1273, down 1090, out 1024;
  at 768 rows 1178 / 1150 / 1224 / 1117. The median launch in the profile (2.409 ms) is 1.21 PFLOPS. Real weights against uniformly random E4M3 bytes cost
  that kernel 3-5 % (1169 vs 1136 TFLOPS at 2048 x 34816), not the 25 % the mean suggests.
- brain's 128 x 256 kernel under the same short, cool run (`fp8_prefill_tflops`, one candidate): gate/up 1033 / 1127 / 1129 at 768 / 2048 / 8192 rows, down
  1153 / 1070 / 1184, qkv 1033 / 1132 / 1125, out 988 / 951 / 1142. So the honest gap is **0-10 % at 8192 rows (largest on down and out, where the reference
  has the K = 17408 / 6144 streams at 1.29-1.30) and 3-14 % at 768-2048 rows**, not 25 %. Two measurements of the same brain kernel differ by 10 % with the
  card's power state (down 8192: 1062 inside a 5-minute sweep, 1175-1184 alone), so a prefill A/B needs interleaved runs.
- The decode figure is real: the 17-64-row gate/up is 58.9 us in the profile (27,085 launches, min 53.2, max 64.3), tile 64 x 144, 7 stages, x as the MMA's
  A operand, the weights as B, 128 math threads.

## Built: `matmul_fp8_wgmma_swap_tma_bn*` (decode, 17-64 rows)

`cu/matmul_fp8_wgmma_swap_tma.cu`: a persistent grid, one thread issuing a 64-row x tile and a BN-row weight tile by TMA into a deep ring (7-11 stages,
as much shared memory as the tile allows), consumer warpgroups taking alternate stages (`bn80`, `bn96`, `bn136`; one warpgroup in `bn136c1`, `bn144c1`),
x as the `wgmma` A operand (no padding at 64 rows), the weight rows as N, a tile cut into `wgmma` of at most 80 columns, the 128-wide scales folded
once per stage (a tile may straddle two weight scale blocks: both are staged, each 8-column group picks its own). BN is any multiple of 8, chosen so the
tile count fills whole waves (34816 = 256 x 136). The shared TMA / mbarrier / tensor-map helpers moved to `cu/fp8_tma_common.cuh`.

Measured (gpu1, `fp8_batch_and_prefill_rate`, weights cycled, us; old = best of the `cp.async` swap kernels, new = best TMA candidate):

| 64 rows | old | new | new TB/s |
|---|---|---|---|
| gate/up 5120 x 34816 | 73.9 | 62.3 (bn96) | 2.86 |
| gate 5120 x 17408 | 37.4 | 36.7 (bn136) | 2.43 |
| qkv 5120 x 10240 | 27.2 | 24.9 (bn80) | 2.1 |
| down 17408 x 5120 | 36.3 | 47.8 | old stays |
| out 6144 x 5120 | 19.3 | 21.6 | old stays |

32 rows gate/up 64.5 -> 61.2. The tuner picks per class, so the old kernels keep down and out.

## What did not move the N = 5120 products (and is not in the tree)

- A cluster of 2 or 4 blocks sharing a tile and splitting K, partial sums added in rank order through distributed shared memory (`bn80s2`, `bn96s2`, `bn160s2/s4`):
  38.5-43 us against the old kernel's 36.3.
- A 2-block multicast of the x tile (each block copies half, the engine delivers to both; consumers arrive on both blocks' empty barriers with the plain
  remote arrive): within 1 % on every shape. The L2 serving the same x lines to 128 blocks at once is not the limit.
- Four independent accumulator chains per stage instead of one: slower (the fold's extra adds cost more than the chain latency they hide).
- A wgmma behind a branch ptxas cannot prove uniform is serialised (rule 16): the loop bounds go through `__shfl_sync`; a pipelined pair of accumulator sets
  with `wait_group 1` trips ptxas's C7514 (non-wgmma reads of accumulators in flight) and ran 20 % slower.

ncu (`--clock-control none`, bn40 at 64 x 17408 x 5120, 43 us, SM 1.82 GHz): tensor pipe 3 % active, DRAM 48 % of peak, L2 52 %; 41 % of warp samples wait on
the full barrier, 8 % on the wgmma, 4 % in the producer's empty wait; a 136-stage tile takes 560 cycles a stage with 195 KB in flight. The consumer is
starved, not busy, and neither ring depth (6 -> 15 stages), scale loads (removed: no change) nor the x multicast changed it, so what limits the small-BN
streams is how fast one SM's TMA completes 128-byte-row boxes, which the reference's 48-row tile (31.6 us) does better; not understood.

## Status: built, gated, NOT selected in serving

The tuner's representative widths (4096 / 16384 at K = 5120) made the large classes pick the old kernels, so the grid's top widths must become the served
ones (10240 / 34816; down 5120) for the TMA candidates to win - that one-line change is NOT landed: with it, `bn96` at 33-64 rows on gpu0 faulted
(`unspecified launch failure`) once per c=64 run in the scoreboard (3 of 3 runs on gpu0, 0 of 2 on gpu1; replacing that one class by the old kernel removed
it; the same run under `compute-sanitizer --kernel-name regex=swap_tma` memcheck finished with 256/256 requests and no report; the unit gate, memcheck,
synccheck and racecheck are clean on both cards at every served shape and row count 17-64). A timing-dependent fault in the persistent ring is open. Racecheck did
flag one real protocol weakness (a warp-leader empty-barrier arrival is not credited for the other lanes' scale loads): arrivals are per thread again.
Serving A/B with the old tuner grid (TMA kernels never chosen) equals base: c1/4/16/64 81.8-82.5 / 256-259 / 562-578 / 836-872 (card noise 2-3 %).
`qwen35_fp8_parity` (8 x 1500 tokens, 32 steps; decode batch 8, so the 17-64-row kernels are not exercised) base vs new: 96.88 % / KL 1.790e-3 vs 96.88 % /
1.796e-3 and 97.66 % / 1.351e-3 vs 97.27 % / 1.354e-3, no flip above margin 0.5.

Training: the FP8 GEMM is not on the NF4 training path; nothing to gradcheck.

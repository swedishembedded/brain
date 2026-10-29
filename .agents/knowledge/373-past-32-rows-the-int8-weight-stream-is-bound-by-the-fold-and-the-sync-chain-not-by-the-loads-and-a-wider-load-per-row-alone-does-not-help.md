<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 373. Past 32 rows the int8 weight stream stays far under the memory roof, and neither a warpgroup kernel, a cheaper fold nor deeper rings moved it

Question: a decode step of 33-128 sequences still reads every weight once (ideal ~7-8 ms of linears on a GH200), yet the
step leaves the weight-streaming candidates at 32 rows and the tiled GEMM reads at 0.8-1.3 TB/s of a 3.68 TB/s roof.
What does a streaming kernel for those rows have to beat?

Mechanism added: the prefill slot (`matmul_i8_prefill`, the rows past `DECODE_REGIME_MAX_ROWS`) gained streaming
candidates for 33-64 rows (`matmul_i8_stream_r*`, the mma.sync kernel of #369 with room for 64 rows of x) and 33-128 rows
(`matmul_i8_wgstream_n{64,128}_s{2,4}`: `wgmma` m64nNk32 with the weights as the A operand and the batch of x as B,
128 weight rows a block, K split over a cluster). The prefill grid's 33-128-row classes are now timed on weights cycled
past the L2 and graded against the memory roof (`Grid::stream_max_rows`), as the decode grid's are. The tuner picks them where they win: K=17408 classes (the down projection,
a quarter of the weights), 25-35 % faster than `matmul_i8_mma_m128`; the K=5120 x N=17408 gate/up classes keep the
tiled GEMM (76 us against 86-97 us, ideal 27 us). End to end (8K context, kv int8): batch 48 51.9 -> 47.6 ms, 64 58.3 ->
53.9 ms, 96 and 128 unchanged. Linears reach 27 % of the roof at 48 rows (24 % before).

What the measurements say (standalone harness over the kernel source, weights past the L2, one card):
- Loads alone, 48 rows, K 5120 x N 17408: the cp.async ring moves 2.1-2.7 TB/s with 128-byte visits to each weight row;
  asking for 256 contiguous bytes of a row per visit raised the loads-only figure to the roof. The compute then hid it
  no better: full kernels stayed at 1.1-1.3 TB/s.
- Taking the MMAs out of the mma.sync stream kernel at 48 rows saves 12 us of 81, taking the whole compute out saves 37:
  the fold (12 FP instructions per MMA per thread), `ldmatrix`, the scale loads and the per-stage `__syncthreads`
  together, not any one of them. The fold bound itself (2 FFMA per output, 128 rows) is ~8 ms of the step, so a kernel
  that streams at the roof up to 128 rows needs the fold at two instructions per output AND the sync chain hidden.
- Not wins (reverted): scales rounded and expanded once per stage in shared memory (knowledge 372's two-FMA fold) in the
  mma.sync kernel - 69 -> 81 us at 48 rows, the extra barrier cost more than the instructions it saved; a second
  accumulator set so the tensor cores run group q+1 during group q's fold in the warpgroup kernel; wide mma.sync blocks (128 weight rows,
  8 warps; and tiles of 96-128 rows of x, which spill); 256-byte stages in pairs with a 6-tile ring (one block an SM, no overlap).
- Occupancy is the lever the mma.sync kernel responds to (3 stages x 4 blocks an SM beat 4 stages x 2: 69 vs 93 us),
  because its chain (ldsm -> mma -> fold) is latency-bound per warp.

Rule: before writing a many-row streaming kernel, state the instruction budget per output per k32 group (MMA, fold,
loads) against 16.5 B/clk/SM and measure loads-only, no-fold and no-MMA variants of the kernel in a harness first; a
block of 128 weight rows on 136 blocks (N 17408) leaves four SMs with two blocks, so tile counts near a multiple of the SM
count also need a cluster split.

Trap: `i8_gemv_native` and the other `gpu-core` card tests that take no `--device` argument open the first CUDA device
(GPU 0 on this node), and `CUDA_VISIBLE_DEVICES=1` makes them fail to build their backend; run them only when the node
is yours.

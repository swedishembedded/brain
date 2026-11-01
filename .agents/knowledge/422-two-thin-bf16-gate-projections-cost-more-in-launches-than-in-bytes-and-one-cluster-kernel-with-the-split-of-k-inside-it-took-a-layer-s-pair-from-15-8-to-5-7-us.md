<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 422. Two thin bf16 gate projections cost more in launches than in bytes, and one cluster kernel with the split of K inside it took a layer's pair from 15.8 to 5.7 us

Question (the brief of 2026-11-01): a Gated DeltaNet layer's `in_proj_b` and `in_proj_a` (48 weight rows each over a 5120-wide row, bf16, 983 KB for the pair) at 16 rows cost 671 us a decode step
(nsys, 48 layers) against vLLM's 317.

## What the 671 us was

At 2..16 rows the dense multi-GEMV (`DENSE_GEMV_MAX_ROWS = 1`) does not serve and each matrix went to the library: `nvjet_sm90_tss_16x64_..._splitK` 5.3 us and `splitKreduce_kernel` 1.7 us, twice a
layer, after an activation cast (`cast_f32_bf16`, 1.7 us): 15.8 us a layer for 983 KB, i.e. 4 launches of the weight's 0.3 us of bandwidth. vLLM merges the two into one `in_proj_ba` (one split-K
product and one reduction: 317 us). The staged 2..8-row multi-GEMV puts 32 weight rows in a block (4 blocks for the pair): a latency-bound walk of the 5120 columns, not an option.

## What shipped

`matmul_bf16_gemv_splitk` (`Fused::Bf16GemvSplitK`): up to four bf16 matrices, 2..16 rows, in ONE launch: a cluster of 8 blocks takes a 16-row tile of weight rows (6 tiles for the pair, 48
blocks) and splits K between them; the tile's weights and the activations go through `mma.sync m16n8k16` (the activations read as f32 and rounded to bf16 in registers, what `cast_f32_bf16` and
therefore the library's product do: no cast launch), the 8 warps reduce through shared memory and the 8 blocks through the cluster's, in ascending order (the same bits every launch). The lane
takes 8 consecutive k of its weight row with one 16-byte load, the B operands of two MMA steps, and the same 8 consecutive k of its two activation rows: the MMA sums its 16 k slots in any
order, so any common assignment of k to slots is the product. The weights do not depend on the kernel before it, so their loads are issued before `griddepcontrol.wait`. The block is small
(256 threads, 9 KiB of shared memory; knowledge 420).

Kernel time (events, 8 rotated weight copies, 5120 wide, pair of 48): 2 rows 6.0, 4 rows 5.8, 8 rows 6.1, 16 rows 6.0 us (the launch floor of an event pair is ~4 us). nsys, 8K x 16 decode-only
steps: the pair 671 us -> 272 us a step (5.7 us a layer); step period median 17588 -> 17085 us (-2.9 %).

## Numerics

Not bit-identical to the library's (the order of the f32 sum differs, the operands are the same bf16 values): the gate holds each output to 1e-5 of the root of the sum of squares of its terms
against the f64 product of the bf16 operands, over m = 2..16, k = 32..8192, one to four matrices; `ops_matmul_group` holds the grouped pair to the separate library products within 1e-4 of the
output's rms. Served greedy decoding is not bit-reproducible run to run (16 concurrent requests, 64 tokens: base against base 12 of 16 identical, new against base 10 and 8 of 16).

## Rules

- A thin projection is priced in launches: count the cast, the split-K product and its reduction before blaming the bytes.
- A cluster kernel that reads a peer's shared memory arrives with release after the peer loads (knowledge 420). `back_to_back_launches_return_the_first_launch_s_bits` runs 3,000 launches at
  2 and 16 rows, a 4096-row matrix and one tile (Xid count unchanged).

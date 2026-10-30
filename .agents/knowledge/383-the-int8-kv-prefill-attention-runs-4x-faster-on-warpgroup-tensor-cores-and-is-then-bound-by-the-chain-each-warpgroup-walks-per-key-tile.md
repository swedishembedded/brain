<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 383. The int8-KV prefill attention runs 4x faster on warpgroup tensor cores and is then bound by the chain each warpgroup walks per key tile

Question: knowledge 382 put 2.4 s of the 6.2 s TTFT of a 31K prompt in `flash_prefill_f16_hd256_v2_kv8` (60-83 TFLOPS causal). How far does a
FlashAttention-3-shaped kernel (wgmma, warp specialisation, split key range) take it, and what limits it?

Result (gpu1 = GH200, `cuda_flash_prefill -- --ignored contexts`: 512-row chunk over a growing int8 cache, 24 heads, head_dim 256):

| context | v2 `mma.sync` | `flash_prefill_f16_hd256_wg_kv8` | |
|---|---|---|---|
| 1K | 0.162 ms (60 TFLOPS) | 0.088 ms (110) | 1.8x |
| 4K | 0.615 ms (79) | 0.219 ms (220) | 2.8x |
| 8K | 1.231 ms (81) | 0.387 ms (258) | 3.2x |
| 16K | 2.464 ms (82) | 0.606 ms (335) | 4.1x |
| 31K | 4.82 ms (82) | 1.088 ms (364) | 4.4x |

Served (FP8 Qwen3.8-27B, int8 KV, same client): 31K-token TTFT 6.2 s -> 4.9 s; the 1K-prompt sweep (c = 1 / 4 / 16 / 64: 52.1 / 214.9 / 365 /
525 tok/s) did not move, a 1K prompt spends 0.1 ms a layer here.

What the kernel is (`crates/kernels-cuda/cu/flash_prefill_f16_hd256_wg.cu`): 384 threads, a producer warpgroup (cp.async of the raw int8/E4M3
bytes into a double-buffered staging area, exact widening to fp16 into 128-byte-swizzled K/V tiles, mbarriers) and two consumer warpgroups of 64
query rows (S = Q K^T as 16 `wgmma m64n64k16`, online softmax on the accumulators, O += P V as 4 `m64n256k16` with P in registers and V by
descriptor, transposed). The row scales are applied to the fp32 scores (K) and to the fp16 weights (V), so the producer's widening is a
byte-to-fp16 conversion with no multiply. A block's key range is split over up to 4 blocks (`wg_eff_splits` picks how many work from the key
count; a combine kernel merges): 96 blocks of 128 rows x one head fill 0.73 of the 132 multiprocessors, split-key blocks fill them.

What was measured on the way (each a one-line change, 10 s per iteration on the kernel's own binary):

1. **Split key range.** Without it (one split) the same kernel runs 1K/4K/8K/16K/31K in 0.070/0.201/0.371/0.705/1.339 ms; with 4 splits
   0.088/0.219/0.387/0.606/1.088. It wins from 16K and loses at 1K/4K because the surplus blocks of a launch sized for the longest context
   still occupy an SM for a block's start (1 block of 231 KB of shared memory each): ~18 us at 1K. The grid is fixed per recorded tape, the
   effective split count is decided on the device.
2. **A `wgmma` behind a per-thread condition is issued one at a time** (kernels.md item 16): ptxas put a `WARPGROUP.DEPBAR` between all 16
   S products; running them unconditionally took the tile from 1.23 to 1.13 ms.
3. **Registers.** `m64n256` needs 158 registers under the launch bound; 512 threads (bound 128) is a ptxas fatal, 384 threads (168) compiles
   and `setmaxnreg` lets the consumers use up to R206 (232 granted, producer 40).
4. **Where the time goes** (ncu + `clock64` per phase, one consumer warp, per tile of 64 keys, 5.2K cycles): K wait 150, S issue 2660, scale
   330, softmax 710, P pack 270, PV issue 980. With the producer's widening removed the kernel is 0.78 ms (505 TFLOPS), with the fetch also
   removed 0.78, with widening but no fetch 1.12: the widening costs 0.33 ms of shared-memory bandwidth and issue slots shared with the
   consumers, the L2 fetch only 0.07 (the fetch alone ran at 3.5 TB/s of L2 reads, the ceiling of the kernel's tile shape, 0.44 ms at 31K).
5. **The small-N product is latency bound.** A probe kernel (one block, 16 `m64n64k16` into one accumulator, repeated): 94 cycles per wgmma for
   one warpgroup, 97 with two at once; `m64n128k16` 127/141; `m64n256k16` (A in registers) 219/299. The peaks are 32/64/128 cycles. The
   tensor pipe is busy 37-40 % of the time at 364 TFLOPS: the work per warpgroup is a dependent chain (S, softmax, PV) and nothing else overlaps it.
6. **Tried, no gain:** a ping-pong token order of the two warpgroups' S and PV phases (as FA3), two independent accumulation chains for S,
   running K one tile ahead of V in the producer, a linear raw staging layout with per-tile pool-row table in shared memory (slower: 1.33 ms).

Not done, in the order the measurements point: (a) intra-warpgroup overlap (issue S of tile j + 1 before the softmax of j; needs ~32 more
registers, i.e. 240 granted), (b) a 128-key S tile (N = 128 is 91 % of the tensor peak against 66 % at N = 64) which needs the K/V tiles
resident for 128 keys, i.e. a smaller Q or staging footprint, (c) multicast of the K/V fetch over a cluster of the 6 heads sharing a kv head
(cuts the L2 traffic 6x), (d) the training forward and backward: this kernel serves the paged inference cache only; the dense training
forward (`BRAIN_DENSE`, fp32 K/V, writes the log-sum-exp) and `flash_attn_bwd_f16` are untouched and unchanged in speed.

Consequence: a prefill chunk's attention at 31K context is 1.1 ms a layer instead of 4.8; the 31K TTFT is now bound by the GDN kernels, the
GEMMs and the row copies named in knowledge 382, and the next attention gain is (a) or (b), not the memory system.

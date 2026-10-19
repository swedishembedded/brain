<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 331. Software-pipelining the shared-x int8 GEMV does not lift its bandwidth at m=4 and m=8

Question: the decode GEMVs reach 2.7-3.0 TB/s at one row of x but 1.3-2.4 TB/s at four and 1.1-1.9 TB/s at eight
(`i8_gemv_native_bench`, 27B shapes), and they are 73 percent of a batch-8 decode step (window6 b8 table, knowledge
327). The rows above two use `brain_gemv_tile_shared_x`: each stage loads weights, fills a shared tile of x, syncs, then
computes, so nothing overlaps across stages. Would issuing stage s+1's weight and x loads before computing stage s lift it?

Result (exclusive card 1, `runs/b27/window10.log`, best of four rounds, GB/s, base / pipelined with 2 vectors per stage):
ffn down 17408x5120 m=8 1806 / 1859, ffn gate/up m=8 1507 / 1555, attn q m=8 1398 / 1457, gdn qkv m=8 1402 / 1419,
ffn down m=4 2351 / 2371, gdn qkv m=4 1863 / 1891. That is +0-4 percent, inside noise. Three and four vectors per
stage were slower at m=8 (ffn down 1594 and 958). The pipelined kernel stayed byte-identical to the WGSL tier
(`i8_gemv_native`), but it was not kept.

So the bound at m>2 is not the load/sync serialisation. What remains to test: the x broadcast reads from shared memory
(8 rows x 16 B per weight vector per thread), the dp4a plus f32 fold arithmetic that the bit-identity contract fixes, and
the 16-lane fold with 32 shuffles per row. Relaxing bit identity with the WGSL tier (a tensor-core m16 path like
`matmul_i8_mma_m16`, which batch 16 already uses at 1.6 TB/s) is the larger lever and a numerics decision.

Trap: several rows in the window logs read 80-400 GB/s for a 30 us kernel; any other process on the card does that to a
device-event timing, so a bandwidth row is only believable when the best of several rounds sits next to a quiet baseline.

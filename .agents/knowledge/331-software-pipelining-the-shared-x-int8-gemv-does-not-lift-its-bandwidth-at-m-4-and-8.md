<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 331. Software-pipelining the shared-x int8 GEMV does not lift its bandwidth at m=4 and m=8

Question: the decode GEMVs reach 2.7-3.0 TB/s at one row of x but 1.3-2.4 TB/s at four and 1.1-1.9 TB/s at eight
(`i8_gemv_native_bench`, 27B shapes), and they are 73 percent of a batch-8 decode step (window6 b8 table, knowledge
327). The rows above two use `brain_gemv_tile_shared_x`: each stage loads weights, fills a shared tile of x, syncs, then
computes, so nothing overlaps across stages. Would issuing stage s+1's weight and x loads before computing stage s lift it?

Result (`runs/b27/gemv-q1.log`, card 1 selected with `BRAIN_DEVICE=gpu1`, best of two rounds, GB/s, base / pipelined with
2 vectors per stage): ffn down 17408x5120 m=8 1865 / 1926, ffn gate/up m=8 1551 / 1607, attn q m=8 1452 / 1525, gdn qkv
m=8 1474 / 1485, ffn narrow m=8 1117 / 1220; m=4 changes by under 1 percent. That is +1-9 percent, not the lever. The
kernel stayed byte-identical to the WGSL tier (`i8_gemv_native`) but was not kept. A dp4a variant that sums each 32-element
group as an int32 before one scale multiply (a third of the arithmetic per weight byte) gained 8-15 percent at m=4 and
nothing at m=8 (`gemv-q1.log`, `grp1`): what bounds m=8 is not the load structure and not the arithmetic of the f32 fold
but the activation operand - every weight byte meets eight activation bytes read back from shared memory, which is the
shared-memory bandwidth at the weights' own rate. That is what a tensor core avoids (knowledge 335).

An earlier version of this entry reported a no-gain result from `window10.log`; those numbers were taken on card 0
while the teacher server ran there (see the trap below) and are withdrawn.

Trap: the plain gate binaries (`i8_gemv_native_bench` and the other non-`card_tests` binaries) pick GPU 0 unless
`BRAIN_DEVICE=gpuN` is set, and the reservation helper does not pin a device - `card --card 1` only reserves shares. A
bandwidth row that reads 80-400 GB/s for a 30 us kernel is another process on the card. Check which card a run used
(`nvidia-smi --query-compute-apps`) before believing a table, and give every bench the device explicitly.

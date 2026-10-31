<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 401. The int8 decode attention was bound by its loads (half of each sector, issued in lockstep with the compute) and by a grid just past a wave

Question (knowledge 400): the 27B's decode attention (4 kv heads x 256, GQA group 6, int8 pool, split-key on f16 tensor cores) took 141 us a layer at 16 sequences x 8K keys for
268 MB of KV bytes, 1.76x the byte time at the 3.4 TB/s streaming roof (`decode_attention_roof_report`, GH200). Where did the time go?

Findings (`ncu --clock-control none` on the test binary, 64 x 8K):

- The kernel held 192 registers and two 4-warp blocks an SM: eight warps, each loading its whole chunk (16 keys, 256 bytes a lane) and only then computing it. Issue slots 35 %
  busy, 5.5 cycles between issues: a latency-bound loop, not a bandwidth-bound one (DRAM 56 %).
- The lane map read 64 contiguous bytes a lane (four 16-byte loads), so one load instruction touched 16 bytes of each of 32 different 32-byte sectors. With the plain loads the
  other half arrived through L1 on the next instruction; a bulk copy that bypasses L1 (`cp.async.cg`) fetched every sector twice (141 -> 173 us). Reading the 16-byte pieces
  lane-interleaved (a lane's piece i at `i * 64 + 16 * lane` of the row) makes one instruction a whole 64-byte (keys) or 128-byte (values) run; the channel order inside a dot
  product is free, so only the query fragments and the merge's channel index move.
- `cp.async` into per-warp shared-memory stages (each lane reads back what it copied, so the only synchronisation is the warp's own `wait_group` and a `__syncwarp` for the
  scales) takes the 64 registers of the staged loads off the critical path of the next chunk. Two stages, three 4-warp blocks an SM (168 registers, 74 KB of stages a block) beat
  three stages and two blocks: more warps hid the dependent chains (issue slots 41 % busy at 8 warps) better than more bytes in flight.
- One division and one block-table lookup a chunk instead of one per row: a chunk's 16 keys lie in one pool block when the block size is a multiple of 16.
- The grid: splits of 8 tiles (1024 keys) made 512 blocks at 16 x 8K, 1.3 waves of 396 resident blocks. Grids cut finer lose to the fixed cost of a block (single-tile splits run
  at two thirds of the speed of eight-tile splits: query fragments, first loads in flight, records), so the cut is chosen by `waves * (tiles + 6)` over 1..32 tiles a split
  (`flash_splits_kv8`).

Measured, us a layer (GB/s of the roof), before -> now:

| sequences x keys | before | now | x the byte time |
|---|---|---|---|
| 16 x 8K | 141.2 | 120.2 | 1.50 |
| 16 x 16K | 264.5 | 211.8 | 1.32 |
| 16 x 32K | 505.0 | 392.8 | 1.22 |
| 16 x 64K | 970.4 | 749.7 | 1.17 |
| 64 x 8K | 508.6 | 398.5 | 1.24 |
| 64 x 16K | 976.3 | 754.8 | 1.18 |
| 64 x 32K | 1896.2 | 1463.8 | 1.14 |

What stays: the kernel retires about 1,250 instructions a chunk (the exact int8 -> f16 widening alone is a permute and a subtract per pair of elements, for keys and for values, and
every probability and query value is split into two f16 parts). It is issue-latency bound at 8-12 warps an SM; 16 x 8K is one partly filled wave and sits at 1.5x. Folding the
-128 bias into one per-head correction would remove the keys' subtracts but costs the precision the hi/lo split buys; not done.

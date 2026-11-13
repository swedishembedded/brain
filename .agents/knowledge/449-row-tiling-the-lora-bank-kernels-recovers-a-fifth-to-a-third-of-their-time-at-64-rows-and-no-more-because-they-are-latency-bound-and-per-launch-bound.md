<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 449. Row-tiling the LoRA bank kernels recovers a fifth to a third of their time at 64 rows and no more, because they are latency-bound and launch-bound

**Lesson:** The per-row adapter kernels of knowledge 444 are not limited by the traffic a row-tiled kernel removes. A
tiled projection and epilogue (`lora_xa_rows_tiled`, `lora_delta_rows_tiled`, `lora_delta_rows_multi_tiled`, selected
as `KernelVariant::RowTiled`) read a quarter of the bytes and are 20-30 % faster at 64 rows and no faster at 8, and
the 27B's adapter decode moves from -61 % / -66 % of base throughput to -55 % / -63 % at 32 / 64 rows (one-slot bank; +13 % at 64 rows, noise at 32). What bounds the
kernels is memory latency at low occupancy and a per-launch floor of 9-12 us, and the step launches 512 of them.

## What was measured

`lora_bank_kernels_cost_at_27b_shapes` (`crates/model/tests/it/lora_bank_rows.rs`, ignored), device time of the
projection plus epilogue from timestamp queries, one GH200, k = 5120 groups of four (qkv z b a) and two (gate up)
linears, a 6144 -> 5120 and a 17408 -> 5120 single linear, rank 32, every row on one slot (us per pair, row-wise ->
row-tiled): group of 4: 8 rows 27 -> 28, 32 rows 53 -> 37, 64 rows 90 -> 60; gate+up: 8: 29 -> 27, 32: 60 -> 46, 64: 108
-> 75; out_proj: 8: 23 -> 28, 32: 28 -> 27, 64: 39 -> 28; down: 8: 38 -> 37, 32: 44 -> 39, 64: 66 -> 41. With four
slots interleaved row by row the tiled pair is 20-120 % SLOWER than the row-wise one at every size (a tile whose rows
run under different slots shares nothing and has a sixteenth of the workgroups), so `lora_bank_rows` offers it only to
a bank with one slot, and `Op::LoraBankProject` from 2048 (row, rank index) pairs, one tile per multiprocessor.

End to end (`brain serve`, Qwen3.8-27B-FP8, `--adapter-slots 1`, greedy, 128 new tokens, base 636 / 1723 / 2800 tok/s
at 8 / 32 / 64 rows): adapter 296 / 771 / 1049 tok/s (-53 / -55 / -63 %); before (knowledge 444, base 614 / 1910 / 2750)
305 / 748 / 930 (-50 / -61 / -66 %): the gain is real at 64 rows (+13 %) and within noise at 8 and 32. Under nsys the
bank's kernels are 38 % of the window's GPU time with the tiled kernels, 42 % before; the rest of the gap to the base is
the 1,500 small launches' own time between kernels.

## What cost the most time

- A function-scope `array<f32, N>` indexed by a loop variable is local memory in the WGSL -> CUDA path: the first tiled
  kernels (arrays of accumulators and loaded values) were no faster than the row-wise ones; the same kernels with
  scalar variables (generated, 16 accumulators and 16 loaded values spelled out) were 1.4-1.6x faster. Write register
  tiles as scalars.
- `ncu` on the kernel's own test (`lora_bank_row_tiled_projection_over_interleaved_slots_for_a_profiler`): 84 registers,
  12 % of the warp slots active (64 workgroups of 256 threads on 132 multiprocessors), 66 % of cycles stalled on the
  long scoreboard (memory latency). A tile divides the workgroup count by 16, so below about 128 tiles it cannot hide
  latency, which is where the 8-row cases lose.
- The epilogue loops now stop at the slot's rank (the rank word's high 16 bits; the low 16 bits stay the live width the
  projection reads), which the group epilogue had not: it ran to the bank's cap, so an adapter of rank 16 in the new
  default cap 32 read zeros for half its loop. Not measured end to end (no rank-16 adapter was served).

## What to do

The next step is structural, not another tile shape: a fused projection-and-epilogue per layer unit with split-k over
workgroups for small batches, or the library tensor-core GEMM per resident slot for the one-adapter case (the rollout
server), or folding the correction into the base FP8 GEMM's epilogue. Each removes launches, which is where the time is.

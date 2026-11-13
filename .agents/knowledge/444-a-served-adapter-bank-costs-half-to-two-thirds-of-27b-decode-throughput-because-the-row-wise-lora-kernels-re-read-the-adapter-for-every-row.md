<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 444. A served adapter bank costs half to two thirds of 27B decode throughput because the row-wise LoRA kernels re-read the adapter for every row

**Lesson:** Knowledge 315's one measurement (a quarter of batch-8 throughput on the GGUF) understated a cost that is not
fixed: on the Qwen3.8-27B FP8 release the bank's overhead per decode step grows with the number of rows, so the throughput
lost grows with the batch. It is a missing kernel variant, not a wrong choice between existing ones.

## What happened

2026-11-13, `brain serve --serve-adapters DIR --adapter-slots 4 --adapter-rank-cap 32 --adapter-targets all
--max-batch 64 --ctx 1024 --reserve-gb 16`, Qwen3.8-27B-FP8, one GH200, three rank-32 adapters on all twelve linears
(934 MB each, hot-loaded), a closed loop of N concurrent chat requests of 128 new tokens, greedy, three rounds, the first
dropped:

| rows | base | all rows on one adapter | rows over base + 3 adapters |
|---:|---:|---:|---:|
| 8 | 614 tok/s | 305 (-50 %) | 308 |
| 32 | 1910 | 748 (-61 %) | 800 |
| 64 | 2750 | 930 (-66 %) | 1040 |

The extra time per decode step is about 13 ms at 8 rows, 26 ms at 32 and 47 ms at 64: about 0.75 ms a row, the time to
stream the 934 MB adapter once per row.

## Why (profiled)

`nsys --trace=cuda --cuda-graph-trace=node`, an 8 s window of 32 rows on one adapter (the decode is graph-replayed, so the
default graph-level trace shows no decode kernels): the bank's three kernels are 3.03 s of the window's kernel time,
about 42 %: `lora_xa_rows` 1.68 s (51,026 launches, 33 us each), `lora_delta_rows_multi` 1.06 s (41 us), `lora_delta_rows`
0.29 s (11.5 us); the base's FP8 GEMMs are the rest. `lora_xa_rows` runs one 64-thread workgroup per (row, rank index):
each re-reads its row's activation (k floats) and one row of A (k floats), so a launch of 32 rows over a stacked group of
128 rank rows moves 164 MB through L2 in 33 us, which is L2 bandwidth, and `lora_delta_rows_multi` re-reads each slot's B for
every row. Rows that share an adapter (every row, in a rollout group) share nothing.

## What to do

Not fixed here. A row-tiled variant of the two kernels (a register tile of 4 rows by 4 rank indices when the rows share a
slot, so each loaded value feeds four products, with the rows' slots checked at run time and the row-wise kernel as the
fallback) selected by rows through `Op::LoraBankProject` would cut the traffic about four-fold; filling the card at 8
rows needs the reduction over k split across workgroups. `lora_delta_rows_multi` also loops to the bank's rank cap, not the
slot's rank (`lora_delta_rows` stops at the rank), so an adapter of rank 16 under the new default cap of 32 reads zeros for
half of its loop; not measured.
A bank narrowed with `--adapter-targets attention` pays for fewer linears.

<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 472. A merge folded into the last split block of a decode is latency bound, and lost to the separate merge launch at every batch

**Lesson:** saving the split-key decode's second launch by letting the block that finishes a (sequence, kv head)
last merge its splits moves the merge onto ONE block per group. That block walks every split of every head of the
group through L2 with a few loads in flight per thread, and the walk takes longer than the separate merge launch
plus its launch gap, at one sequence and at 64. The cost of a merge is its parallelism, not its launch. Cutting the
separate merge's split sum over four threads a channel did not help either: the launch already had enough loads in
flight, and the extra barrier and shared-memory partials made it slower.

## What happened

The goal was batch 1 at 32K keys, where the two-launch decode (split pass, then `paged_flash_decode_gqa_combine`)
reads the pool at 55-65 % of the 3.4 TB/s roof on a GH200 (nsys, Qwen3-4B int8: 28.5 us split plus 4.2 us merge a
layer). A one-launch candidate kept every block's partial records and a completion counter per (sequence, kv head)
behind them. The last block to arrive re-read the group's records with `__ldcg`, merged them in split order and
re-armed the counter. It was exact (same bound against f64 as the chain) and launch-to-launch bit-stable, and slower
everywhere (wall time of a submit, best of 25, gpu3):

| shape, tier | batch 1 chain / one launch | batch 64 chain / one launch |
|---|---|---|
| 32/8 x 128, int8 | 42.7 / 51.5 us | 1408 / 1491 us |
| 24/4 x 256, int8 | 48.1 / 89.7 us | 1470 / 1587 us |
| 16/2 x 256, bf16 | 48.0 / 111.7 us | 1391 / 1557 us |

A group of 6 or 8 heads of 256 with about 85 splits is about 0.5 MB of records for one block. With 8 loads in
flight per thread and an L2 round trip of about 0.6 us, that block streams them at single-digit GB/s.

The separate merge with four threads a channel (eight partial sums each, then a fixed-order add in shared memory)
came out no faster at batch 1 and doubled at 64 rows (`decode_attention_matrix_cell_report`: 12-15 to 20-28 us).

## What to do

- Batch-1 decode attention needs the split pass itself nearer the roof (it is the 28 of the 33 us: one wave of short
  blocks with too few warps a scheduler to hide the chain latency), not a cheaper way to merge. A merge has to stay
  spread over at least as many blocks as the separate launch uses.
- Measure the folded form against the chain before building on it. The evidence and the attempt's diff are kept
  outside the repository: they were measured on this commit's kernels and did not land.

## Evidence

GH200 gpu3, 2026-11-16: `decode_attention_one_launch_report` (the attempt's own timing test) for the table above,
and `decode_attention_matrix_cell_report` for the four-lane merge. The one-launch candidate passed
`native_decode_of_every_tier_is_exact_at_the_qwen3_head_shapes_on_cuda` and its re-armed-counter repeat check.

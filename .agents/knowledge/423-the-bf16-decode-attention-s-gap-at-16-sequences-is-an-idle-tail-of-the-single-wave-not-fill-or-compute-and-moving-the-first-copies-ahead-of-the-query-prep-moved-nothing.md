<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 423. The bf16 decode attention's gap at 16 sequences is an idle tail of the single wave, not fill or compute, and moving the first copies ahead of the query prep moved nothing

Question (the brief of 2026-11-01): `paged_flash_decode_gqa_hd256_kvbf16` is 189 us a layer at 8K x 16 (nsys, served) against vLLM's 155 us for the same 537 MB (3.46 TB/s); knowledge 415 put it at 81 % of the
byte roof and blamed the single wave. What is the loss made of?

## Measured (ncu on the kernel's own test binary, `decode_attention_matrix_cell_report`, 16 x 8192, 256 blocks of 128 threads)

- 186 us, DRAM 72 % of peak sustained (2.91 TB/s), L2 83 %, L1/TEX 35 %, tensor pipe and issue slots nearly idle (compute throughput 19 %, one warp in five eligible): the kernel is memory-bound where it runs.
- `sm__cycles_active` over the 132 SMs: **avg 280 k, max 311 k, min 182 k cycles of 316 k elapsed**. 256 blocks on 132 SMs at two a SM leave 8 SMs with one block, which finish at 58 % of the time of the
  SMs with two (a block alone moves 19 GB/s, two on an SM 24 GB/s each: the SM, not the block, is what saturates), and the SMs with two end between 280 k and 311 k cycles. A mean 13 % of the card
  idles while the last blocks finish: 186 us x 0.87 = 162 us, the byte time at ~3.3 TB/s. The loss is the tail, not the fill (the first copies were already in flight before the query fragments were built:
  issuing them first changed nothing, 185.7 -> 185.7 / 187.5 us) and not the tensor work.
- 64 x 8192 and 16 x 32768 (512 blocks, two waves of 264 slots with the wave structure absorbing the spread) reach 3.1 TB/s (90 %).

## What would close it (not done)

A static cut cannot: the spread is in how fast blocks run (memory partitions, neighbours), and `flash_splits_kv8`'s model already prices waves and a block's fixed cost (6 tiles; the fixed cost is the
query prep, the pipeline fill, the four-warp merge and the record write). The loss is recovered by letting a late SM take work from a slow one: a tail of each block's range (say its last 2 of 16 tiles) as
a separate unit with its own partial record, taken by whichever block finishes first through an atomic ticket (a counter that the last block of the launch resets, so a replayed graph needs no memset);
the combine reads the extra records (it merges any count). Expected: up to the 13 %, less the extra query prep and merge of a stolen unit (~3 %). A flat cut (264 blocks, each a range of equal
tiles over the whole batch's (pair, tile) space, crossing pairs) would fill the 8 idle slots and balance ragged batches, which the cut from the longest sequence does not.

## Rules

- Read `sm__cycles_active` max / min / avg before calling a single-wave kernel "latency bound": here avg / elapsed was 0.89 and min / max 0.59.

<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 416. An f32 GDN state streams at 95 % of the roof at 64 sequences, but at 16 the block structure and the ramp cap it near 76 %, and the served step does not move

Question (the brief of 2026-11-01): the f32-state recurrence (`gdn_recur_pool`, 38.4 us a layer at 8K x 16 served) against the reference server's 28.4 us over the same
100 MB of state. Can the bf16 stream kernel's design (bulk copies, two columns a thread) reach the f32 state's roof?

## What was built

`gdn_recur_pool_stream` (after the unchanged `gdn_conv_pool`) and its verify form `gdn_recur_verify_pool_stream`: a block of 128 threads a (sequence, value head), the 64 KiB f32
tile in eight bulk-copy chunks with a barrier each (the first pass starts on a chunk as it lands, each chunk is stored as soon as its rows are final), a thread owning two adjacent columns
over half the rows. The conv cannot join the head blocks: it rewrites the q and k windows that the 3 head blocks of a key head share, so one block per key head (192 KiB of tile, one block an
SM) was the alternative and was not built. Both conv kernels take the fused qkv|z row stride as a fourth param; the recurrence reads z in place after the qkv columns.
Verify is the same body over `counts[bi]` rows, so a verified row is bit for bit the plain streaming step (an f32 pool with the occupancy candidates on streams in both; the exact body
verified against a streaming decode would drift).

## Numbers (gpu1, `gdn_decode_pool_production_shape`, 27B head counts, event time of the recurrence kernel; the pool holds 4x the batch's rows so L2 does not keep the state)

| sequences | exact `gdn_recur_pool` us | stream us | state traffic | % of 3.4 TB/s |
|---|---|---|---|---|
| 1 | 12.1 | 9.4 | 6.3 MB | 20 |
| 4 | 18.4 | 14.5 | 25 MB | 51 |
| 16 | 48.6 | 39.2 | 101 MB | 76 |
| 64 | 139.0 | 124.6 | 403 MB | 95 |

nsys of the same bench (no event overhead): 34.3 us at 16, 120 us at 64 (3.36 TB/s). The 28.4 us target is not reached.

## Where the time goes

- A copy-only version (load, store, no arithmetic) took 35.7 us event time at 16 against 39.2 for the full kernel: the arithmetic is ~3 us; the structure is the limit. A loads-only version took
  62 us at 64 sequences (201 MB, 3.24 TB/s): the bulk-copy read side alone does not exceed ~3.3 TB/s.
- A fixed ~3.5 us ramp (first bytes, last store) is 12 % of the 16-sequence roof and 3 % of the 64-sequence one.
- Tried and no gain: eight chunk barriers with the first pass starting on landed chunks and per-chunk stores (34.3 -> 34.5 us at 16); 256-thread blocks with four row splits (34.4, 120). Neither
  changed the time, so it is bound by the memory pipeline, not the arithmetic or the ordering inside a block.
- The bench's 20 us gap between conv and recurrence is host launch; in the served graph the recurrence's tile loads overlap the conv through the programmatic launch.

## Served (FP8 / bf16 KV / f32 state / no speculation, gpu1, two interleaved rounds)

Not distinguishable from noise: the layer saving is ~5 us x 48 layers = 0.25 ms of a 50 ms step. See the ledger for the table. Speculation (MTP) at 8K x 1 is noisy (controller): new 80.1 / 81.9 / 92.4,
base 88.4 / 88.7 / 90.7 tok/s over three runs each.

## Rule

An f32 state pool is bound at about 3.3 TB/s by bulk-copy reads; to reach the reference's 3.5 the read path would have to change (per-thread vector loads in flight, not TMA), which the bf16 work (knowledge 396)
found slower for lack of in-flight bytes. Judge a state kernel by sequences >= 64 for the roof and by the served step for value.

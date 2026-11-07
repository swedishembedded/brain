<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 405. Widening the 8-bit KV cache once per round instead of once per query block takes the prefill attention from 390 to over 500 TFLOPS

**Lesson:** Doing a conversion once per round into a staging buffer, instead of repeating it inside
every query block, removes a producer bottleneck and lets the consumers keep the tensor pipe busy. A
trick that did nothing while one stage was the bottleneck (warpgroup ping-pong) can pay once that
stage is fixed.

## What happened
Question: knowledge 383 left the head_dim 256 warpgroup prefill (`flash_prefill_f16_hd256_wg_*`) at
364-390 TFLOPS causal with the tensor pipe 40 % active, one 384-thread block per multiprocessor, and
its producer warpgroup the suspect (the widening alone cost 30 % of the kernel there). How much of
the gap to a FlashAttention-3 class kernel (served 64K x 1 prompt: 5.64 s of attention against about
3.5 s for the reference server) is the producer?

Result (gpu1 = GH200, `cuda_flash_prefill -- --ignored --nocapture long_prompt_round`, 4096-row
round, 24 heads, 4 kv heads, int8 cache; the rate includes the widening pass):

| context | in-block widening (`wg`) | widened once, 64-key tiles | 128-key tiles | 128-key tiles, warpgroups ping-pong |
|---|---|---|---|---|
| 4K | 301-311 TFLOPS | 390-405 | 421-446 | 446-465 |
| 8K | 346-355 | 465-487 | 496-532 | 527-546 |
| 16K | 354-381 | 494-531 | 522-540 | 559-566 |
| 24K | 367-388 | 506-547 | 532-560 | 570-578 |
| 32K | 390-394 | 507-549 | 518-543 | 567-601 |

Ranges are repeated runs on the same card; it is power capped and its clock moves between runs by up
to 15 %, so a comparison is made inside one process or as best-of-three.

A 512-row chunk gains too: 4K / 8K / 16K / 31K context 0.219 / 0.388 / 0.610 / 1.088 -> 0.176 /
0.313 / 0.520 / 0.920 ms. ncu at 32K, 4096 rows: tensor pipe active 40 % -> 62 % (64-key tiles) ->
73 % (128-key tiles with the ping-pong).

## Why
The producer was the bottleneck: each block converted its keys itself. Two further effects set the
rate of the wider tiles:

- S = Q K^T is then 16 `wgmma m64n128k16` whose dependent chain costs 127-141 cycles an instruction
  against 94-97 for `m64n64k16` (knowledge 383 item 5), 91 % of the tensor peak against 66 %.
- The per-tile fixed work (barriers, row maxima, rescale vote) is paid per 128 keys.

The ping-pong pays on the 128-key form (+5-8 %, 73 % tensor active). Both warpgroups of a block
start every tile together, so without a turn order their softmaxes coincide and the tensor pipe
idles. With the order S0, S1, PV0, PV1 (named barriers 4-7, slot i waited on by warpgroup i % 2 and
arrived at by the warpgroup of slot i - 1, slot 0 primed by warpgroup 1) one warpgroup's softmax
runs under the other's products. Knowledge 383 tried it while the producer was the bottleneck, where
there was nothing to gain.

## What to do
What changed:

- `flash_prefill_f16_hd256_wgs_widen_{kv8,kvfp8}` widens the round's keys once per layer into a
  staging buffer of one record per (kv head, 64-key tile): the K and V tiles in the exact 128-byte
  swizzled fp16 layout the tensor cores read, and their row scales (K scale already times 1/sqrt(d)
  log2 e) beside them. Keys past the longest sequence are zero with zero scales.
- `flash_prefill_f16_hd256_wgs` is the version 3 consumer with its producer warpgroup reduced to one
  thread issuing `cp.async.bulk` copies of whole tiles (no raw staging, no per-block conversion).
- The first form kept 64-key tiles in two-stage rings; the second takes 128-key tiles, one stage
  each (64 KiB K, 64 KiB V, 198 KB of shared memory).
- The numerics are version 3's bit for bit (the widening is exact, the scales are applied to the
  fp32 scores and to P as before); the specs in `cuda_flash_prefill.rs` run both forms and print
  identical differences.

Cost and limits:

- The pass reads the context's 8-bit bytes and writes 2 bytes per element plus scales: at 32K
  context 200 MB, about 60 us against a 6 ms attention.
- The staging buffer is `n_kv_heads x ceil(keys / 128) x 132096` bytes a layer (270 MB at 64K, 4 kv
  heads), allocated per step like the split scratch.
- Rounds of fewer than `FLASH_PREFILL_STAGED_MIN_ROWS` rows keep the in-block form (their attention
  is too short to amortise the pass).
- `PrefillShape::keys` carries the round's longest key count to size the buffer.

Tried on top, no gain (each a one-line change on the 64-key form, measured in the same session):

- An intra-warpgroup overlap (issue S of tile j + 1 and P V of tile j back to back, softmax of j + 1
  under the product; 53 % tensor active, 440 TFLOPS): the dependent S chain in front of P V in the
  in-order tensor queue costs more than the overlap hides, and ptxas serializes every wgmma (C7514)
  unless the consumers get 240 registers and the overlapped region has no condition around a wgmma
  or its wait.
- Two independent accumulation chains for S (-5 %).
- The ping-pong turn order of the two consumer warpgroups on the 64-key form (S0, S1, PV0, PV1; -3
  %).

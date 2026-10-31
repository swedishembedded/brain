<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 405. Widening the 8-bit KV cache once per round instead of once per query block takes the prefill attention from 390 to over 500 TFLOPS

Question: knowledge 383 left the head_dim 256 warpgroup prefill (`flash_prefill_f16_hd256_wg_*`) at 364-390 TFLOPS causal with the tensor pipe 40 %
active, one 384-thread block per multiprocessor, and its producer warpgroup the suspect (the widening alone cost 30 % of the kernel there). How
much of the gap to a FlashAttention-3 class kernel (served 64K x 1 prompt: 5.64 s of attention against about 3.5 s for the reference server)
is the producer?

Result (gpu1 = GH200, `cuda_flash_prefill -- --ignored --nocapture long_prompt_round`, 4096-row round, 24 heads, 4 kv heads, int8 cache; the
rate includes the widening pass):

| context | in-block widening (`wg`) | widened once (`staged`) |
|---|---|---|
| 4K | 301 TFLOPS | 390-405 |
| 8K | 346 | 465-487 |
| 16K | 354 | 494-531 |
| 24K | 367 | 506-547 |
| 32K | 390 | 507-549 |

(the spread is two runs; the card is power capped and a neighbour on the other card moves the clock). A 512-row chunk gains too: 4K / 8K / 16K /
31K context 0.219 / 0.388 / 0.610 / 1.088 -> 0.188 / 0.318 / 0.530 / 0.969 ms. ncu at 32K, 4096 rows: tensor pipe active 40 % -> 62 %.

What changed: `flash_prefill_f16_hd256_wgs_widen_{kv8,kvfp8}` widens the round's keys once per layer into a staging buffer of one record per
(kv head, 64-key tile): the K and V tiles in the exact 128-byte swizzled fp16 layout the tensor cores read, and their row scales (K scale already
times 1/sqrt(d) log2 e) beside them; keys past the longest sequence are zero with zero scales. `flash_prefill_f16_hd256_wgs` is the version 3
consumer with its producer warpgroup reduced to one thread issuing `cp.async.bulk` copies of whole tiles into a 2-stage K ring and a 2-stage V
ring (shared memory 232 KB -> 194 KB, no raw staging, no per-block conversion). The numerics are version 3's bit for bit (the widening is exact, the
scales are applied to the fp32 scores and to P as before); the specs in `cuda_flash_prefill.rs` run both forms and print identical differences.

Cost and limits: the pass reads the context's 8-bit bytes and writes 2 bytes per element plus scales: at 32K context 200 MB, about 60 us against a
6 ms attention; the staging buffer is `n_kv_heads x ceil(keys / 64) x 66048` bytes a layer (270 MB at 64K, 4 kv heads), allocated per step like the
split scratch. Rounds of fewer than `FLASH_PREFILL_STAGED_MIN_ROWS` rows keep the in-block form (their attention is too short to amortise the pass).
`PrefillShape::keys` carries the round's longest key count to size the buffer.

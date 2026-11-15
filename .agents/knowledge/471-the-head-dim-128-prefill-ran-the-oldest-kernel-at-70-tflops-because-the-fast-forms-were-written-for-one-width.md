<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 471. The head_dim 128 prefill ran the oldest kernel at 70 TFLOPS because the fast forms were written for one width

**Lesson:** A kernel family optimised at one shape leaves every other shape on whatever it had before, and nothing
reports it: the slow width still passes its parity tests and is "on tensor cores". When a variant is tuned, make the
width (or tier) a compile-time parameter of the fast source in the same change, and register it at every width the
old form served, or list the width as a gap.

## What happened

A Qwen3-4B serve (`perf run serve`, 16 x 8K prompts, gpu3 GH200, nsys) spent 41 % of its kernel time in
`brain_flash_prefill_f16_hd128_kv8`, 7.87 ms a call for one 8K causal prompt of 32 heads: about 70 TFLOPS. That kernel is
version 1 of `flash_prefill_f16_hd256.cu` compiled with `BRAIN_HD 128`. The warpgroup (`wgmma`) forms and the staged
form (knowledge 405, 434), at 450-600 TFLOPS on the same card, existed only at head_dim 256: their sources hard-coded the
256-wide tile sizes, the 16-step S chain, the 128 output registers and the `m64n256k16` P V product, and the selection
declined any other width.

## What changed

- `flash_prefill_f16_hd256_wgs.cu` (the widen pass and the attention) derives every width-dependent count from `HD`:
  tile and record bytes, the Q tile, the `HD / 16` S steps, `HD / 2` output registers and an `m64n128k16` P V product at
  128. `flash_prefill_f16_hd256_wg.cu` compiles its helpers and the combine kernel alone (`BRAIN_WG_HELPERS_ONLY`) at any
  width; its own kernel stays head_dim 256 over an 8-bit cache.
- The widen pass gained the fp32 tier (`BRAIN_KV_TIER 0`, rounded as version 1 rounds it, unit row scales, read by the
  unit-scale attention of knowledge 434), so every tier the old kernel served at 128, and the fp32 tier at 256, has the
  staged form.
- `paged_flash_prefill_wg_steps` takes head_dim 128 and 256 and every tier; at 128 every round is staged (there is no
  in-block form, and the per-call kernel is 3-5x slower at every row count measured). The split scratch is allocated only
  for rounds that split, and a caller that records many rounds into one tape can share one `WgPrefillScratch`.
- Version 1 at head_dim 128 stays as the kernel of parts without warpgroup tensor cores (cc 8.0-8.9).

## Evidence

`cuda_flash_prefill_hd128::the_qwen3_prefill_rates` (gpu3, 32 heads, 8 kv heads, causal TFLOPS best of ten / sustained
over 400 rounds; the staged figure includes the widen pass):

| round | tier | version 1 | staged warpgroup |
|---|---|---|---|
| 8192 rows, no context | int8 | 77 / 76 | 403 / 358 |
| 8192 rows, no context | bf16 | 83 / 82 | 485 / 398 |
| 8192 rows, no context | fp32 | 90 / 85 | 463 / 376 |
| 4096 rows behind 12K | int8 | 75 / 74 | 427 / 380 |
| 512 rows behind 3.5K | int8 | 72 / 72 | 253 / 253 |
| 512 rows behind 15.5K | int8 | 67 / 67 | 330 / 318 |
| 512 rows behind 31.5K | int8 | 66 / 66 | 359 / 334 |
| 512 rows behind 31.5K | bf16 | 71 / 71 | 426 / 382 |

ncu at the 8K int8 round: 2.19 M cycles, tensor pipe 50 % active, 372 M warp instructions, 6.6 cycles a warp between
issued instructions. At head_dim 256 the same kernel reaches 73-83 % (knowledge 405, 434): a 128-wide head halves the
products per score while the softmax per score stays, so the non-tensor-core work is twice the share.

Tried, no gain: a two-stage K/V ring at head_dim 128 (the smaller tiles leave room for it): 414 / 363 against 403 / 358 at
the 8K int8 round and 370 / 332 against 359 / 334 at 512 x 31.5K, within the card's run-to-run spread. The copy of the
next tile is not what the consumers wait on; the ring was not kept.

Gates: `the_warpgroup_prefill_at_head_dim_128_reads_every_cache_tier` and the large-score-magnitude and back-to-back
(knowledge 410) runs at both widths, every tier, against the portable kernel and an f64 oracle.

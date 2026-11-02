<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 434. A bf16 prefill attention spent a fifth of its instructions on per-key scales that were all the same, and the power-limited card paid for them in clock

Question (brief of 2026-11-02): raise the SUSTAINED (400 rounds) rate of the bf16 staged prefill attention (`flash_prefill_f16_hd256_wgs`) at 4096-row rounds over 4K to 64K contexts, toward a FlashAttention-3-class kernel.

## Measurements (gpu0, 24 heads, 4 kv heads, hd256, causal TFLOPS, `the_bf16_cache_prefill_rates_at_the_long_prompt_round`)

- Sustained clock: the sustained phase of the staged kernel runs at 1395-1470 MHz at 510-530 W (the v2 kernel in the same test sits at 1980 MHz, 480 W). Best-of-ten is a cold-clock number; the sustained one is what a served prefill gets.
- ncu at locked clocks (32K context, 4096 rows): old kernel 7.86 M cycles, tensor pipe 73 %, 1.09 G warp instructions; vLLM's FlashAttention-3 `device_kernel` 6.82 M cycles, tensor 84 %, 1.32 G instructions. FA3 executes MORE instructions and still takes 13 % fewer cycles: instruction count is not the cycle limit.
- The old kernel's SASS per 128-key tile and warp: 64 FMUL (K scale), 64 FADD (score minus reference), 64 FMUL (V scale), 32 LDS.64 (the scale vectors), 64 FADD (row sum), 64 FMNMX, 64 MUFU. For a bf16 cache the staged K scale is the constant 1/sqrt(d) log2 e and the V scale is 1 (dead keys weigh 0 through the mask), so 128 multiplies and 32 shared loads were pure overhead.
- The attention kernel is shared by all tiers (the scales travel in the staged record), so a first try that only guarded `BRAIN_KV_TIER == 1` changed nothing (SASS identical): the bf16 variant is its own entry, `flash_prefill_f16_hd256_wgs_bf16` (`BRAIN_WGS_UNIT_SCALE`). The raw scores are masked and maximised unscaled, the running maximum lives in the scaled domain (a row with no live key keeps NEG_BIG), and the exponent is `ex2(fma(s, scale, -ref))`.
- After: 858 M instructions, 6.93 M cycles, tensor pipe 82.7 % (FA3: 6.82 M, 84 %), shared-memory wavefronts 574 M -> 480 M.

| ctx (4096 rows) | before best / sustained | after best / sustained | FA3 sustained (no staging) |
|---|---|---|---|
| 4K | 443 / 389 | 488 / 409 | 530 |
| 8K | 558 / 455 | 616 / 493 | 520 |
| 16K | 607 / 484 | 666 / 527 | 543 |
| 32K | 628 / 498 | 685 / 544 | 562 |
| 64K | 609 / 511 | 696 / 558 | 567 |

Our figure includes the widen pass (100 us of 5.5 ms at 32K, 1.8 %); FA3's is the attention alone. The 4K gap is fixed overhead (widen, the two-kernel dependency, block prologues over 1 to 32 tiles), not the inner loop.

## Open

- The int8 and FP8 tiers keep per-key K and V scales: the K scale must scale the scores before the maximum, so the multiply cannot fold into the exponent; they do not gain from this.
- Every round re-widens the whole context (1.8 % at 32K); keeping staged records across rounds would remove it.
- Thread-block clusters with multicast of the K and V tiles to blocks of one kv head would halve L2 to shared traffic (energy per FLOP); not tried.

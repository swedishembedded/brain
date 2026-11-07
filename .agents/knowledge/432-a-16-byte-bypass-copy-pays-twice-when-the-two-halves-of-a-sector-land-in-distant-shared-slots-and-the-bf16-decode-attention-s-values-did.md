<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 432. A 16-byte bypass copy pays twice when the two halves of a sector land in distant shared slots, and the bf16 decode attention's values did

**Lesson:** An asynchronous copy of a wide row fragment pays twice when the two halves of a sector
land in distant shared slots. Read the sector counters of the profiler instead of trusting a
bandwidth figure, and place the halves adjacent in shared memory.

## What happened

Question (the brief of 2026-11-02): brain trails vLLM at 8K x 64 by about 4 s of attention in an 80
s window. Where, and what is the largest part?

### Where the attention time goes (whole-run nsys, 8K x 64 bf16 KV, exclusive time)

- brain: decode 6.16 s (2.1-2.2 GB a call, 705 us, 3.05 TB/s), prefill 4.21 s + widen 0.22 + merge
  0.20, qk prep 0.55, KV append 0.50, decode merge 0.12: 11.99 s. vLLM: `device_kernel` 8.95 s
  (decode and prefill rows together), qk/rope 0.72, cache write 0.14: 9.79 s.
- The byte budget is 2.15 GB a 64-row call (not 1 GB): 625-705 us is 3.05-3.4 TB/s. vLLM reaches
  about 3.65 TB/s on the same rows, so the roof is above 3.4.
- Prefill attention runs at the card's sustained clock, not the best trial: staged bf16 n=3968
  ctx=7892 is 1.054 ms best of ten but 1.289 ms over 400 back-to-back rounds (547 -> 447 TFLOPS),
  which is what the served 1300-1380 us calls show. It is energy-bound like the FP8 GEMM (knowledge
  411); no kernel change was tried.
- The bf16 KV append was the generated kernel: a read-modify-write of a pool word per value, 55 us
  for a 4096-row prefill round (150 GB/s). The native `kv_append_bf16` does it in 16 us (1.57 TB/s
  of its moved bytes), bit-identical on ties, NaN, infinity, subnormals.

## Why

### The finding: ncu on the 64 x 8K decode kernel

- DRAM 77 % of peak (3.10 TB/s), but L2 throughput 89.6 % and L2 to L1 bytes 3.22 GB for 2.15 GB of
  DRAM reads: every value sector was returned twice.
- Probes (ncu, same binary): keys only 1.08 GB DRAM, 1.08 GB L2->L1; values only 1.08 GB DRAM,
  **2.15 GB** L2->L1, 634 us (keys only 390 us); values fetched like keys 1.08 GB, 368 us.
- Cause: the value copy gave eight adjacent lanes 128 contiguous bytes of a row, but the lanes'
  shared slots (`lane * 16`) put the two halves of a 32-byte sector four slots apart. Changing only
  which lane fetches what (128-byte runs, consumer-slot destination) changed nothing; the halves
  must also be ADJACENT in shared memory. Source: fetch like the keys (eight rows an instruction, 64
  contiguous bytes read by four adjacent lanes) and put sector halves side by side
  (`brain_value_slot`); the consumers read their slot after the existing warp wait and `__syncwarp`.
- ncu "L2 Theoretical Sectors Global" per source line (32 against 16 ideal for the value copies)
  shows this in one table.

## What to do

### The 8-bit kernels had the same value copy

The int8 and FP8 kernel (`paged_flash_decode_gqa_hd256_kv8.cu`) got the same fetch. int8 / FP8 us a
layer, before -> after: 16 x 8K 119.8 / 119.3 -> 109.7 / 108.4; 16 x 32K 402 / 396 -> 364 / 363; 64
x 8K 403 / 401.5 -> 364.5 / 365.3; 64 x 32K 1486 / 1483 -> 1342 / 1348; 64 x 64K 2905 / 2900 -> 2640
/ 2638 (97 % of 3.4 TB/s); 1 x 8K 24 us. Same specs and 2,000 back-to-back launches, unchanged.

### Rules

- Read ncu's "L2 Theoretical Sectors Global" and "Excessive" per instruction before trusting a
  bandwidth number: L2 to L1 bytes above DRAM bytes is a load pattern, not a memory-system limit.
- A `cp.async` copy of 16 bytes a lane must put the two lanes that share a 32-byte sector in
  adjacent shared slots; the consumer's layout is free to follow.

## Evidence

### Kernel time after (gpu0 idle, split pass + merge, us a layer; before -> after)

| rows x keys | before | after |
|---|---|---|
| 1 x 8K | 27.7 | 26.5 |
| 4 x 8K | 65.3 | 59.6 |
| 4 x 32K | 194.0 | 176.3 |
| 4 x 64K | 365.0 | 324.4 |
| 16 x 8K | 194.8 | 176.0 |
| 16 x 32K | 698.0 | 630.9 |
| 16 x 64K | 1420 | 1273 |
| 64 x 8K | 705.5 | 635.8 |
| 64 x 32K | 2771 | 2486 |
| 64 x 64K | 5405 | 4863 |

16 x 8K is 3.05 TB/s (90 % of 3.4), 64 x 64K 3.53 TB/s. Exact: the decode specs (ragged, scattered,
shared blocks, NaN-poisoned records, empty splits) and 2,000 back-to-back launches at 16 x 8K, 64 x
1K, 1 x 32K per tier are unchanged.

### Served A/B (gpu0, bf16 KV, f32 GDN state, no speculation, two interleaved rounds, base = the build before the change, tok/s)

| cell | base | new |
|---|---|---|
| 8K x 16 | 317.8, 316.4 | 320.4, 297.3 (one slow round) |
| 8K x 64 | 380.9, 371.6 | 382.8, 383.4 |
| 32K x 16 | 88.5, 89.0 | 88.2, 88.5 |
| 64K x 4 | 37.2, 37.6 | 37.6, 37.3 |

FP8 KV (one round each, only the bf16 kernel and the append changed, so the arm is a control): 8K x
64 405.0 -> 403.2, 32K x 16 91.7 -> 91.5. The 64-row gain is +0.5 to +3 %, within the card's
round-to-round spread: decode attention is about 6 % of the step, so 10 % of it is 0.6 %. The 12 %
throughput gap to vLLM is not in the attention kernels: the GPU idles 7.6 s of an 80 s window
against vLLM's 2.7 s.

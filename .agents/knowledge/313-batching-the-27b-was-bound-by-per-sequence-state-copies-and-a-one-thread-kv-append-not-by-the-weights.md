<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 313. Batching the 27B was bound by per-sequence state copies and a one-thread KV append, not by the weight reads

Question: serve the released Qwen3.8-27B (Q8_0 GGUF, INT8 on one GH200, CUDA) to many
sequences per GPU step through the one generic continuous-batching scheduler, and find what
limits aggregate decode throughput. Idle card throughout (`other_contexts_start=0`), ctx 4096,
512-token prompts, greedy, `make perf/model/qwen35-batch` (`qwen35_batch_sweep`). All figures are
**VERIFIED-BY-RUN** unless marked **DERIVED**.

## What the profile found, in the order it was removed

Per-kernel device time of one decode step (`qwen35_decode_profile`, synthetic context 1024, bf16 KV):

| kernel family | batch 64 as first measured | after pooling GDN state | after chunked KV append | after small-tile GEMM |
|---|---:|---:|---:|---:|
| `splice` + `concat_split` (per-sequence state staging) | 120.9 ms (51%) | 0.2 | 0.2 | 0.2 |
| `gdn_state_decay`, `bmm`, `bmm_acc`, `l2norm` (19-kernel GDN chain) | 62.1 ms | - | - | - |
| `gdn_decode_pool` (one native launch per layer) | - | 12.7 | 12.8 | 12.8 |
| `paged_kv_append_batched_word` (bf16) | 7.1 | 6.8 | 0.2 | 0.2 |
| int8 GEMM (`matmul_i8_mma`) | 31.5 | 29.4 | 29.3 | about 25 (**DERIVED** from the per-shape sweep) |
| step total, device | 236 | 58 | 51 | about 47 |

1. **The weights were not the bound.** Each sequence's recurrent state and conv history were
   copied into a contiguous slab and back with one `splice` per sequence per layer, and the
   step then ran the 19-kernel GDN chain on that slab. At 64 sequences those copies plus the
   chain were three quarters of the device time. The state is now a per-layer pool addressed
   by slot (as `qwen35moe` already did), so a batch runs on its rows in place in the native
   `gdn_decode_pool` launch; prefill works on a private copy that is scattered into the row.
2. **`paged_kv_append_batched_word` stored a 1024-wide row from ONE thread** (to keep each
   packed bf16 word private to a thread): 6.8 ms of a step at 64 sequences. A thread now owns
   a 16-element chunk; words are still private, the row spans 64 threads: 0.2 ms.
3. **The 64x64-tile tensor-core GEMM starves a 132-SM part at decode row counts.** 16-128 rows
   give it 80-272 blocks and it streamed 0.75-1.4 TB/s of weights. The same source at 16x16,
   32x32 and 64x32 tiles (chosen by row count) reaches 1.5-1.8 TB/s at 16 rows (1.7x on the
   whole 27B shape set), 1.4x at 32, 1.16x at 64, 1.1x at 128. Bit-identical results. At 64+
   rows the kernel is bound by the f32 fold of the group scales, not by bandwidth.

## Aggregate decode, ctx 4096, 512-token prompts, one GH200 (device wall per step, host included)

| sequences | bf16 KV tok/s | ms/step | tok/s per sequence | KV + state GiB |
|---:|---:|---:|---:|---:|
| 1 | 76.6 | 13.0 | 76.6 | 0.40 |
| 4 | 220 | 18.2 | 55.1 | 1.58 |
| 8 | 299 | 26.7 | 37.4 | 3.17 |
| 16 | 594 | 26.9 | 37.1 | 6.34 |
| 32 | 965 | 33.2 | 30.2 | 12.7 |
| 64 | 1385 | 46.2 | 21.6 | 25.4 |
| 96 | 1473 | 65.2 | 15.3 | 38.0 |
| 128 | 1716 | 74.6 | 13.4 | 50.7 |

int8 KV: 64 / 128 / 192 sequences = 1255 / 1706 / 1863 tok/s (17.5 / 35.0 / 52.4 GiB). f32 KV: 64 / 96 =
1206 / 1407 tok/s (41.4 / 62.0 GiB). The sampling head (top-k extraction) costs nothing
measurable: bf16 at 64 sequences 1194 tok/s sampled against 1182 greedy in the same build.
Before this work the served path decoded about 36 tok/s (one sequence at a time).
Weights take about 29 GiB, so the planner's ceiling at 4096 tokens on one 97 GiB GH200 with a
2 GiB reserve is 103 / 168 / 244 sequences at f32 / bf16 / int8 KV (pinned by a unit test;
128 bf16 and 192 int8 were run, the ceilings themselves were not).

## End to end is prefill-bound for short outputs

Serving 32 concurrent requests of a 512-token prompt and 64 new tokens through the resident
(`--e2e`) takes 12.1 s, 169 tok/s generated: prefill is about 1700 tokens/s (0.3 s per
512-token prompt), 9.6 s of the 12.1. Decode rates above apply once the prompts are in.
That is why prefix reuse matters for agents: a resumed prompt prefills only its new tokens
(real 27B, 381-token prompt, 0.88 s cold against 0.19 s resumed, same continuation).

## What batching does and does not change

Batched output equals the single-sequence reference token for token on the real checkpoint
(CUDA and Vulkan), and a seeded sampled sequence does not depend on its neighbours. An
earlier build (staged state) diverged on one of four prompts on CUDA after two tokens: the
batched and the one-row GEMM round differently, so a near-tie can flip. Greedy equality is
therefore an observed property of that gate, not a guarantee of the arithmetic.

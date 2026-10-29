<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 377. Dequantizing int8 weights in registers removes the fold and wins from 33 rows and at prefill; below that the load structure caps it under the int8 stream

Question: #372 and #373 put the Q8_0 linears of Qwen3.8-27B under the group-scale fold: prefill GEMM at 15-23 % of the int8
tensor-core roof, decode linears at 1.1-1.3 TB/s above 32 rows of a 3.7 TB/s read roof. If the weight code is multiplied by its
group scale once, as it leaves shared memory, and the activations stay bf16, the tensor core accumulates across all of K with no
fold. What does that buy, and where?

What was built (`cu/matmul_i8_dq_stream.cu`, `cu/matmul_i8_dq_wgmma.cu`; the candidates declare `Input::DEQUANT_BF16`, #376):

- bf16 weights from int8 codes in about 2.5 instructions per element, exactly: `prmt` spreads two bytes into 16-bit lanes,
  `(u & 0x7F) | 0x4300` is the bf16 `128 + low7`, `(u & 0x80) | 0x4300` is `128` or `256` (bit 7 is the exponent's low bit),
  their difference is the signed code (a small integer, exact in bf16), one bf16 multiply by the group scale (rounded to bf16)
  gives the weight. Relative error of a weight against `code * scale`: at most about 2^-8.
- Streaming kernel (rows 33-128): the `matmul_i8_stream` ring (cp.async, ldmatrix, K split over a cluster) with `mma.m16n8k16`
  bf16, weights as A (16 rows), rows of x as B (8 a tile). The MMA's reduction order is free, so k is permuted inside each
  16-deep step to match what `ldmatrix` hands a thread (kernels.md C.15); the activation fragment is one eight-byte load.
- Hopper GEMM (prefill, rows past 128): `wgmma` m64n128k16 with both operands from shared memory, activations bf16 as A, the
  dequantised weights as B written by all 256 threads into three 128x128-byte tiles while the tensor cores run the previous
  k-tile (the wgmma of tile kt is issued first, tile kt + 1 is dequantised behind it).
- Dispatch: a candidate declares the activation form it reads; `Gpu::int8_linear_bf16_input` asks the tuned table, `Ops` hands
  the bf16 rows of `Act` (one `cast_f32_bf16`, shared by every product that reads them) to `Gpu::int8_dequant_step`, and
  `Ops::act` skips the `max_abs_row`/`quant_pack` pair when every class of that row bucket and width reads bf16 (a stray
  product that still needs the packed form makes it then). No model code changed.

Measured (GH200 gpu1, idle; tool: the autotuner's class table, `i8_dequant_native` for device time, `ncu` for the why):

| shape (rows x K x N) | best int8 candidate | dequantize candidate |
|---|---|---|
| 48 x 17408 x 5120 | `wgstream_n64_s4` 96 us | `dq_stream_r4` 56 us |
| 48 x 5120 x 17408 | `mma` 75 us | `dq_stream_r4` 60 us |
| 96 x 17408 x 5120 | `wgstream_n128_s2` 167 us | `dq_stream_q2` 116 us |
| 1024 x 5120 x 17408 | `wgmma` 706 us (258 TOPS) | `dq_wgmma` 577 us (316 TFLOPS) |
| 4096 x 5120 x 5120 | `wgmma` 834 us (258 TOPS) | `dq_wgmma` 651 us (330 TFLOPS) |
| 4096 x 17408 x 5120 | `wgmma` 2710 us | `dq_wgmma` 2422 us |
| 8..32 x 5120 x 17408 | `stream_s1` 36 us (2.7 TB/s) | `dq_stream` 45-54 us (2.0-2.2 TB/s) |

- From 33 rows the streaming kernel wins 14-55 % of the time of the best integer candidate (17-31 % for 65-128 rows), 16 of 16
  classes at 33-64 rows. The warpgroup GEMM wins about 0.8x the time from 129 rows, except K = 17408 with N 512-1536 at 129-512 rows
  (1.5x slower: four 128-row weight tiles leave the card idle; the tuner keeps `mma` there).
- Below 33 rows nothing won, in eight variants (cluster splits 1-8, 64 and 128 weight rows, 4 and 8 warps, 3-4 stages): they all stop
  at 2.0-2.2 TB/s. The same kernel with the dequantisation and the MMAs removed (loads only) reaches 2.3-2.5 TB/s, so the limit is the
  load structure, not the arithmetic: bf16 activations double the x tile the ring copies (as many bytes as the weights at 32 rows).
  Not registered; the integer streaming set keeps those classes.
- A bf16 `mma.sync` GEMM with the same dequantisation in the MMA warps (not registered) reached 213 TFLOPS with the activations as A
  and 242 with the weights as A (an `IMAD.MOV` per A register went away): under the int8 `wgmma`'s 268 TOPS, because the dequantisation
  and the loads share the issue slots of the warps that issue the MMAs (ncu: tensor pipe 40 %, issue slots 41 %, `LOP3` 20 % of the
  instructions). The warpgroup kernel moves the arithmetic behind asynchronous `wgmma` and reaches 355-372 TFLOPS; without the
  dequantisation arithmetic at all it reaches 388-408, so what is left is the staging (shared-memory bandwidth of the two warpgroups'
  operands plus the copies), not the dequantisation.

End to end, `qwen35_decode_profile --context 8192 --kv int8`, `qwen35_gguf_prefill_profile 256 1024 3`, `brain perf run longctx`
and `brain perf run sweep` (same seed), base -> new:

- `qwen35_decode_profile --context 8192 --kv int8` ms/step, three runs of each build (batch 48 / 64 / 128): 48.0, 47.2, 47.9 ->
  40.3, 39.8, 41.5 / 53.3, 53.1, 53.3 -> 49.8, 48.8, 49.5 / 92.5 -> 88.0. Batches of 32 and fewer are the same kernels in both builds
  and moved by less than the run-to-run spread, except 20 rows, which read 27.7-28.6 ms before and 28.9-29.8 ms after in all three
  runs (+4 %; the same candidates win, the autotuner's own timing of those classes differs by up to 10 % between two tunings; not
  attributed).
- `qwen35_gguf_prefill_profile 256 1024 3`: round 300.5 -> 274.5 ms (3408 -> 3730 tok/s); linears 207.7 -> 171.6 ms, the
  `cast_f32_bf16` 5.2 ms new.
- `brain perf run longctx` (8K, kv int8): prefill 1K / 8K / 32K 3348 / 3682 / 3175 -> 3669 / 3860 / 3308 tok/s; decode 48 / 64 / 128 rows
  1010 / 1201 / 1378 -> 1200 / 1310 / 1475 tok/s.
- `brain perf run sweep --workload chat --ladder 1,4,16 --requests 48 --output 128 --seed 1`: output tok/s 58.5 / 100.0 / 116.2 -> 59.8 /
  102.9 / 128.2, inter-token p99 43.2 / 67.7 / 75.1 -> 41.7 / 66.4 / 69.1 ms, TTFT p99 470 / 3051 / 15088 -> 451 / 4012 / 13537 ms,
  goodput 42.7 / 2.1 / 0 -> 44.9 / 2.1 / 0. A second run at concurrency 4: 102.2 -> 105.0 tok/s, TTFT p99 3191 -> 2834 ms, inter-token
  p99 61.9 -> 63.7 ms: the TTFT p99 at concurrency 4 is queueing behind prefill and spreads by more than 1 s between identical runs, so
  the 4012 ms is not a regression the data supports; the throughput gain (+2-3 % at 4, +10 % at 16) repeats. Goodput does not move: no
  concurrency above 1 met the SLO before or after (the sweep reports the workload exceeds this configuration).

Driver-native evidence (nsys for the timeline, ncu for the kernels; reports kept outside the repository):

- nsys, decode batch 48 (`--cuda-graph-trace=node`, four profiled steps): GPU kernel time of the linears `mma_m128` + `wgstream_n64_s4`
  + `stream_r4` 288 ms -> `dq_stream_r1/r2/r4` 221 ms (-23 %), the whole capture 495 -> 429 ms; with every kernel drained alone the
  GPU is busy 59 % -> 54 % of the steady region, the shorter kernels leave the launch gaps a larger share (the real tape replays as one
  graph). Prefill (all rounds): linears `wgmma` 836 ms -> `dq_wgmma` 731 ms, capture 1212 -> 1073 ms, `cast_f32_bf16` 16 ms.
- ncu on the real model, decode batch 48, same shapes before and after: a 37.7 MB weight class `stream_r4` 49.8 us (0.76 TB/s, 19 % of
  DRAM peak) -> `dq_stream_r4` 30.1 us (1.24 TB/s, 31 %); a 102 MB class `mma_m128` 102.6 us (1.00 TB/s, 25 %) -> `dq_stream_r1` 75.3 us
  (1.37 TB/s, 34 %); ncu pins the clocks, so the absolute figures sit under the autotuner's.
- ncu on the real model, prefill round: `matmul_i8_wgmma` tensor pipe (imma) 15.5 % active, 329 us for 1024 x 5120 x 1536;
  `dq_wgmma` tensor pipe 38-40 % active, 276 us for the same shape (-16 %); issue slots 39 % -> 21 %.

Open: the decode classes below 33 rows (a loads path that does not copy x per block: B fragments from L1, or x resident for small
K); a fused rmsnorm that writes the bf16 rows and no packed rows (the fused front ends still write both, and a `cast_f32_bf16`
costs 5 ms of a 275 ms prefill round); a 128 x 256 warpgroup tile for the shared-memory bound; rows 1-4 (the exact-decode contract
needs one split and one summation order for all of them).

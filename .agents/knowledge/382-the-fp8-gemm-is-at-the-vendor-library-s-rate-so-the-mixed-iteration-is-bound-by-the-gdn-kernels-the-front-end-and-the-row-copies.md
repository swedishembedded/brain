<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 382. The FP8 GEMM is at the vendor library's rate, so the mixed iteration is bound by the GDN kernels, the front end and the row copies

Question: after knowledge 381 the card is busy 90 % of a mixed iteration (145 ms, ~130 ms of kernels). Is the FP8 GEMM (43 % of it) worth a
block-scaled cuBLASLt candidate or a stream-K variant of `matmul_fp8_wgmma`?

Measured (gpu1, FP8 Qwen3.8-27B, chat 1K/128 c=16, nsys, the 78 iterations of 130-170 ms; torch 2.13 `scaled_mm`, which calls
cuBLASLt with the 1x128 activation / 128x128 weight scale modes, timed inside a CUDA graph on the same card):

1. **The library has the modes** (cuBLASLt 13.0 header: `MATMUL_MATRIX_SCALE_VEC128_32F`, `BLK128x128_32F`) with constraints brain's
   layout does not meet as is: the activation scales must be `[K/128][M]` (outer-dim-major, brain writes `[M][K/128]`) and `M` a multiple of 4.
2. **It is not faster where the time is.** `matmul_fp8_wgmma` takes 55.6 of the 130 ms in 400 launches; by grid: 816 blocks 207 us x 113
   (N = 34816 fused gate/up, 187 GFLOP = 900 TFLOPS), 240 blocks 126 us x 119 (down projection, N = 5120 K = 17408, 744 TFLOPS), 480 blocks
   119 us x 44. cuBLASLt on the same shapes at 512 rows: N = 34816 212 us (862 TFLOPS), N = 5120 K = 17408 134 us (680 TFLOPS), N = 17408 98 us
   (929). It wins by 10-25 % only on N = 17408 at 640-1024 rows and loses on the K = 17408 shapes. Against the tuner's table, the classes at 0.5-0.6 of the
   tensor-core roof are the large-N ones; the small-N classes (0.27, ~23 us, 8 tiles on 132 SMs) carry almost no FLOPs. The expected gain of the
   whole integration (new FFI, scale layout, padding, tuner candidate) is 6-8 ms of 145, and a stream-K kernel competes with an already 900 TFLOPS kernel.
3. **Where the other 75 ms are** (ms of kernel time per iteration): `gdn_chunk_loop_tc` 15.9 (188 launches), `gdn_chunk_prep_tok_tc` 8.8,
   `quant_epilogue` 7.2 + `add_rms_quant` 3.2 + `quant_e4m3_128` 2.6, `concat_split` 5.9 (1227 launches) + `splice` 2.9 + `region_copy` 1.7 + pool
   gather/scatter 3.2, flash prefill 4.5, `gdn_conv_silu_tok` 3.1, `mul` 2.8, rmsnorm 2.1. The GDN pieces run 3.9 times a layer in a median
   iteration: every prompt segment (and its short remainder piece) is its own conv, prep, recurrence, gather and scatter launch, each latency
   bound at 50-85 us.

Consequence: the next wins are a multi-segment GDN prefill (one launch for all pieces of a round: grid over segments x heads with the
sequence boundaries as data), the fused e4m3 front end, and row views for the 1227 `concat_split` copies; the GEMM is not the bound.

Single-stream decode (same note, tried and not kept): the decode graph is 18.8 ms a token, 12.3 ms of it `matmul_fp8_dq_stream_r4` (390 launches, ~2.1 TB/s;
ncu at the 17408-wide shapes: 768 blocks of 128 threads, 74 KB dynamic smem so 3 blocks (12 warps) an SM, 1.94 waves, DRAM 1.62 TB/s under ncu).
Four candidates sized for a decode step of at most 8 rows (x tile of 8 rows instead of 64, so a stage costs 10 KB not 24 KB: `m8r1s6`, `m8r2s6`,
`m8r4s6` with six stages, `m8r2s4`) were registered and tuned on the real card: none beat `r4` on any class (rows 1-8, K 5120 and 17408: `r4`
0.59-0.66 of the read roof on the large classes, `m8r4s6` 0.54-0.68, the others 0.3-0.5), and the client c = 1 rate went 50.1 -> 48.3 tok/s (noise
to slightly worse). The stream is not limited by shared memory or ring depth. Per-kernel durations inside the decode graph are inflated by
programmatic dependent launch (a kernel counts from its early start, 24.1 ms of kernel time against 17.6 ms of union busy time).

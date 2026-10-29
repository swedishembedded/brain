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

The int8 packing of the fused front ends (same note, landed): `add_rms_quant` and `quant_epilogue` wrote the int8 rows and group scales of every
layer's activations although no weight of an FP8, bf16, NF4 or f32 build reads them (the unfused `ops_act` already knew: `reads_float_acts`). A flag
(`add_rms_quant` flags bit 1, `quant_epilogue` flags bit 0) now skips the max, the scales, the byte exchange and the stores, and `Qwen35` hands back the float
`Act` (`packs_activations`); mode 0 of the epilogue, which would write nothing, is no launch. FP8 27B, c = 1 / 4 / 16 / 64 output tok/s 50 / 203 / 360 / 500 ->
53 / 213 / 376 / 520, ITL p99 at 16 216 -> 205 ms, 31K TTFT 6.64 -> 6.36 s; `qwen35_fp8_parity` identical (98.44 %, KL 1.526e-3). The training forward does not use the fused
front ends, so the training rate is unchanged by this commit (no regression: the flag is off by default and an int8 build packs as before).

The GDN across-chunk recurrence (same note, landed): ncu on `gdn_chunk_loop_tc` for a 2048-row prompt (32 chunks, 192 blocks on 132 SMs): 1.18 ms, 37 us a
chunk, issue slots 18.6 % busy, 1.45 active warps a scheduler, 7.2 cycles between issues, 161 M warp instructions: latency bound, too few warps
to hide the dependent mma/shuffle chain. The block now has eight warps (four 16-row tiles of the chunk, each split over two halves of the block's 32 columns, same
per-element accumulation order so the result is bit-identical) and `__launch_bounds__(256, 2)` (a first try without the bound spilled to 148 registers, one block an
SM, and gained nothing: 1.23 ms): 0.77 ms (-35 %), 2.85 active warps. Mixed iteration (nsys): the recurrence 15.9 -> 11.0 ms, with the quant_epilogue change above the
median iteration 145 -> 134 ms, card busy 89 %. The training forward and the `train` entry share the body; `gdn_chunk_fwd_train_tc`, `gdn_chunk_fwd_fast`,
`gdn_chunk_fwd_train_fast`, `gdn_mixer_tc/stream/segmented`, `gdn_chunk_bwd` pass; the W8K NF4 train probe stays at 963-968 tokens/s (the recurrence is a small share of a training step).

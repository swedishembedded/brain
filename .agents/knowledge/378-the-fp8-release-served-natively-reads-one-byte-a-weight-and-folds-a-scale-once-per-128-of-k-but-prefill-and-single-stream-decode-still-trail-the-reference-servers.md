<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 378. The FP8 release served natively reads one byte a weight and folds a scale once per 128 of K, but prefill and single-stream decode still trail the reference servers

Question: `Qwen/Qwen3.8-27B-FP8` stores E4M3 weights with a 128 x 128 scale per block (`weight_scale_inv`). Held as stored and
fed to the shared candidate seam (#376), what do the int8 tier's machinery and a Hopper E4M3 wgmma give, and where does the
engine stand against vLLM and SGLang on the same checkpoint?

What was built (candidates declare their input forms, no model code changed):

- Residency: E4M3 bytes plus the f32 block scales, 1 byte a weight (25.5 GiB for the 27B with its bf16 head); `Ops::fp8_forward`
  asks `Gpu::fp8_linear_act` which activation form the tuned slot `matmul_fp8_native` chose for the shape and hands the
  candidate the bf16 rows (one `cast_f32_bf16` shared by the products of an activation) or the E4M3 rows with a scale per 128
  (`quant_e4m3_128`, shared the same way).
- A stage of the int8 streaming kernel (128 bytes of K) is exactly one scale block, so `matmul_fp8_dq_stream_{r1,r2,r4,q2,q4}`
  (1-128 rows) copy ONE scale per stage and multiply it in once per weight pair: E4M3 byte b becomes the bf16 pattern
  `s << 15 | (b & 0x7f) << 4` = decode(b) * 2^-120 exactly, one bf16x2 multiply by `scale * 2^120` gives the weight.
- `matmul_fp8_dq_wgmma` (bf16 operands, weights decoded in shared memory) and `matmul_fp8_wgmma` (e4m3 x e4m3 `wgmma
  m64n128k32`, the four k32 products of a 128-stage accumulated unscaled and folded once with `sx[m,b] * sw[n/128,b]`).
- All seven pass the input-resolution admission (`fp8_linear_native`, ragged n, row counts at every tile edge); the tuner takes
  the e4m3 GEMM for the prefill classes (36 of 96) and `r4`/`q4`/`r1` streaming kernels for decode.

Quality, real checkpoint, 2 code prompts of about 1000 tokens, 24 teacher-forced positions, against the same weights decoded to
bf16 (`qwen35_fp8_parity`): greedy agreement 100 %, mean logit rel-L2 2.3e-2, worst 5.9e-2, mean KL 1.1e-3 nats, no flip at a
reference margin above 0.5. The tiny-fixture conformance (`hf_fp8_conformance`) needs codes of bounded magnitude: with
full-range random E4M3 codes the residual stream has a dynamic range at which bf16 activation rounding turns a batch-shape
difference of upstream f32 sums into another greedy token (the tile decode, which reads f32 rows, hid it).

Measured (GH200 gpu1 idle, kv int8, `brain perf run longctx`, context 8192; int8 = the Q8_0 baseline of serving-runtime.md):

| | FP8 | int8 (Q8_0) |
|---|---|---|
| single stream | 59.3 tok/s (16.9 ms/step) | 72.5 |
| batch 64 / 128 decode | 1337 / 1863 tok/s | 1310 / 1475 |
| prefill 1K / 8K / 31K | 4989 / 5482 / 4439 tok/s | 3669 / 3860 / 3308 |

`brain perf run sweep chat` out tok/s at concurrency 1 / 4 / 16: 52.9 / 128.5 / 181.7 (int8 59.8 / 102.9 / 128.2). Through
`brain serve --openai` and `vllm bench serve` (1024 in / 128 out, random prompts, warm): output tok/s 46.3 / 117.3 / 165.6 /
172.9 at concurrency 1 / 4 / 16 / 64, ITL p50 18.7 / 19.2 / 40.1 / 40.1 ms, TTFT p50 372 ms at 1; vLLM 0.32 on the same card
79 / 299 / 759 / 1241 (ITL 11.2 ms, TTFT 186 ms), SGLang 86 / 214 / 781 / 1044.

Where the time goes (nsys, ncu):

- Decode, batch 1: 13.2 of 20.6 ms of kernel time is `matmul_fp8_dq_stream_r4` (400 launches) = 1.8-1.9 TB/s of the ~3.7 TB/s
  roof (ncu: 46 % DRAM throughput, 110 registers, 74 KB of shared memory a block so 3 blocks = 18.75 % occupancy, 6.7 cycles per
  issued instruction). A decode-sized build (8-row x tile, 41 KB, 5 blocks per SM) was chosen by the tuner for some classes and
  moved the step by nothing (17.25 against 16.87 ms), so occupancy is not the limit; the existing float GEMV
  (`matmul_fp8_gemv_m1`, 20.3 ms) is far worse. 3.7 ms of the step is the add-rms-quant, quant-epilogue and bf16-cast kernels.
- Prefill 8192 (nsys): the e4m3 GEMM is 606 of about 1500 ms at 700 (K = 5120, n = 17408) to 1070 TFLOPS (K = 17408), 35-54 % of
  the e4m3 roof; with only 8-16 row tiles of 128 rows (a 1024-row round) it falls to 25-35 % tensor utilisation (ncu: 171
  registers, 136 KB shared memory, one block an SM, 12.5 % occupancy). The rest is the GDN chunk kernels (300 ms), prefill flash
  attention (142 ms), the int8 quantisation the fused front ends still produce for no consumer (130 ms) and launch gaps (a
  third of a 1K forward).
- vLLM's 31K prefill of 12.8K tok/s needs all of these: the GEMM alone at our rate is 2.3 s of its 2.43 s.

A defect found on the way: a client that sends text without whitespace (vLLM's random prompts) held the serving lane for
seconds in the BPE merge loop (quadratic in the piece length, GPU idle, 19 tok/s total); the loop is now a heap over a linked list.

Left: a bandwidth-bound m <= 8 FP8 GEMV (16-byte loads straight to registers, `cvt` to bf16x2, no shared-memory ring), a
persistent warp-specialised e4m3 GEMM with an epilogue that overlaps the next tile, one fused rms-norm -> e4m3 kernel that
replaces the int8 front ends, and the prefill scheduling that lets a 1K prompt interleave with decode (items 8 and 11).

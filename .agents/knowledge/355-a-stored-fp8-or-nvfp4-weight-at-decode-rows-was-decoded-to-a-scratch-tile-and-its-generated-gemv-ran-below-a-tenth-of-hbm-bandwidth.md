<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 355. A stored FP8 or NVFP4 weight at decode rows was decoded to a scratch tile, and its generated GEMV ran below a tenth of HBM bandwidth

Question: why did the first decode of nvidia/Qwen3.6-35B-A3B-NVFP4 (FP8 mixers, NVFP4 experts and head, served as stored) take
11.5 ms per token for 4 of its 40 layers when the weights of those layers are 0.15 GB?

Cause, in three layers. (1) `Ops::matmul_rows` sent every FP8 / NVFP4 product, whatever the row count, through the tile
decode: at one activation row the whole decoded weight (the 248k-row head is 1 GB of bf16) was written and re-read per token
(`weight_dequant_bf16` was 58% of device time). (2) The first float-read GEMV, the WGSL kernel, was translated to CUDA by
`wgsl-cuda`, which builds without fused multiply-add, has no vector loads, and zero-initialises every `var<workgroup>` table at
entry: the NVFP4 expert gather took 18.6 ms for 12 calls before its decode table moved from a per-call local array to shared
memory (8x), and the head 3.7 ms for 0.29 GB. (3) Even as a native kernel the head stayed at 0.53 ms (550 GB/s) because the
decode, not the bytes, is the cost: 6 integer instructions per weight.

Fix: `matmul_fp8_gemv` / `matmul_nvfp4_gemv` / `moe_nvfp4_gemv_gather` take up to 8 rows straight from the stored bytes, with native
twins (16-byte streaming loads, an FP8 or E2M1 code placed in the top bits of an f16 is the value times a power of two, so the
decode is two masks and a shift and the factor folds into one multiply, warp-shuffle row sums) and the fused add + RMSNorm launch
the int8 tier uses. Rule: a quantized tier gets its decode-regime kernel when it gets its loader; profile the first run per kernel
before reading its token rate, and expect a decode-bound (not byte-bound) kernel to need fewer instructions per weight, not more
loads in flight. A WGSL kernel with two barriers is `@cpu no`: `Ops` must also check the device's `workgroup_reductions` before
dispatching it, or the CPU backend silently runs it wrong (stream_train_packed caught this).

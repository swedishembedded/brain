<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 578. A split kernel a model never registered left its training on the serial fallback, and the weight gradients off the tensor cores

**Lesson:** a shared step builder that picks a fast variant "when the `Gpu` registered it" falls back without a word
when the model's kernel list does not carry it. qwen3's full fine-tune never listed `rmsnorm_dw_part` /
`rmsnorm_dw_sum`, so `model::block::rmsnorm_dw_steps` ran the one-thread-per-channel `rmsnorm_dw` (a serial loop over
every row) for every norm gain, and `Qwen::proj_bwd` computed every weight gradient with brain's fp32 `matmul_dw_reg`
while the forward and the input gradient already ran on the vendor library's TF32 tensor cores.

## What happened
The first nsys profile of Qwen3-TTS training (Talker 0.6B + code predictor as one model, 32 utterances and 4 096 code
predictor frames a micro-step, one GH200 shared with another tenant) put 67.8 % of GPU kernel time in
`brain_rmsnorm_dw` (683 calls, 31 ms each: 16 384 Talker rows and 65 536 code-predictor rows walked serially per
channel) and 21.1 % in `brain_matmul_f32_dw_reg` (1 271 calls, 5.2 ms each). Kernel time for 12 steps and the initial
loss probes was 31.4 s.

## Why
Both fast paths existed and were tested: the two-stage gain gradient by `model/tests/it/rmsnorm_dw_split.rs`, the
TF32 weight gradient by `Ops::matmul_dw`. Neither test can see whether a model's own list registers the kernels or its
own backward calls the shared op, and the fallback is numerically correct, so no parity test fails either.

## What to do
Register the split kernels in the model's kernel list, and route an fp32 weight gradient through `Ops::matmul_dw` when
the GEMM provider is active (the fp32 reference tier keeps the exact kernel for gradient checks). After both, the same
profile's kernel time was 8.1 s (3.9x less), with no kernel above 12.4 %: TF32 GEMMs, the attention and the
elementwise passes. When a profile shows a reference kernel at the top, check the kernel list before the kernel.

## Evidence
`nsys profile -t cuda,nvtx` of `qwen3tts train --steps 12 --sequences 32 --rows 512 --tokens 8192 --frames 4096` on a
6.8 h replicated dataset, before and after the two changes (`cuda_gpu_kern_sum`).

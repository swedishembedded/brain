<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 342. The LoRA rank products ran on fp32 CUDA cores and cost a small model's step 8 percent

Symptom: after knowledge 341 the Qwen3-0.6B LoRA step (3731 tokens, rank 16, auto gemm tier) still spent 40 ms of 255 ms
of kernel time in `sm80_xmma_gemm_f32f32_f32f32_f32_*_ffma` / `cutlass_80_simt_sgemm_*` kernels: the rank-16 products of
`model::adapter::exec` (x A^t, mid B^t, and the five backward products per adapted linear), 2700 launches per step at
about 20 us each on skinny shapes.

Cause: `LibraryGemmProvider::gemm_f32` hard-coded `tf32: false` ("no operand is rounded"), so a tensor-core tier ran the
base weights on tensor cores and the adapter products beside them on the FFMA pipes.

Fix: `F32Product::tf32` (the caller says whether operands may be rounded to TF32; fp32 accumulation either way) and
`AdapterRuntime` sets it. The pinned `Fp32Reference` tier is unchanged: its provider is inactive, `gemm_f32` returns false,
and the serial fp32 kernels run. Kernel time per step 254.9 -> 234.8 ms with no SIMT kernel left (nsys, shared card,
`runs/b27/nsys/lora06b` before and `lora06c` after). The training-signal gates (`lora_gemm_precision` in qwen3 and qwen35,
`qops_frozen_lora_grad`, `lora_learning_gate`, `lora_resume_gate`) hold at their existing tolerances;
`the_tf32_f32_product_stays_within_operand_rounding_of_the_oracle` pins the product's own error (below 3e-3 of the result rms,
and different from the exact form).

What is left: the rank products are 9-30 us tensor-core kernels now, so fusing those that share an input (q/k/v, gate/up)
would save launches, not bandwidth; it needs the A matrices contiguous and is not worth the parameter-layout change at
this size. The step is flash attention (108 ms, knowledge 341) and the base TF32 GEMMs.

<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 336. Chat fine-tuning picks the native bf16 tier when it fits, and brain has no fp8 training path yet

Rule (user, 2026-10-20): training uses the accelerator's native floating-point tensor-core tier by default - fp8 where the
card supports it (Hopper), else bf16/fp16; a quantised base (int8, NF4) only when there is no native tier AND the model does
not fit otherwise.

What `ChatFineTune` does now (`crates/sdk/src/finetune.rs`, `choose_base_tier`). With none of `bf16_base`, `int8_base`,
`nf4_base`, `f32_base` set, the base tier is chosen from the selected cards and recorded on the outcome
(`base_tier`, `base_tier_reason`, and the run record): no selected card at compute capability 8.0 or above (a CPU run, an
older or non-CUDA card) keeps fp32, the exact reference, as before; otherwise bf16 when the bf16 base plus the activations of
a block-token row fit (the largest single card for `qwen35`, whose trainer is one card; the cards together for `qwen3`, which
pipelines), else the quantised base (NF4 for `qwen35`, int8 for `qwen3`). The activation estimate is calibrated on the 27B
(78 GiB at 16K tokens against 67 GiB at 8K for the bf16 base: 1.4 MiB per token at 64 layers of width 5120, scaled by layers
times width) and is a planning estimate, not a measurement of a model it was not fitted to; total card memory is what it
compares to, not what other jobs have left free. Explicit flags win and the old precedence (nf4 over int8 over bf16) holds.
Not changed: the command-line trainers (`brain qwen3 finetune`, `qwen35_*`) keep their explicit flags and defaults.

fp8 today: none. brain reads the 27B's FP8 release (`qwen35::fp8_source` dequantises the checkpoint's e4m3 blocks while
loading) and nothing computes in fp8: there is no fp8 GEMM kernel, no cuBLASLt fp8 call and no e4m3/e5m2 operand path in the
trainers; the int8 serving GEMM and the bf16 training GEMM are the tensor-core tiers brain has. So bf16 is the Hopper default
now.

Plan for fp8 (follow-up, not started): (1) a capability-selected e4m3 GEMM for the frozen base linears of the trainers
through the provider seam (cc 9.0, cuBLASLt or `mma` e4m3 with per-tensor or per-block scales, fp32 accumulation), forward and
the input gradient, the base stored as the checkpoint's own fp8 blocks (27 GB for the 27B, no dequantise-at-load, room for
long blocks), activations quantised dynamically per row, LoRA factors and optimiser in fp32; (2) a measured accuracy gate
against the bf16 reference at the kernel and at the first-step gradient and a 10-step loss curve on the real 27B, as for every
tensor-core tier; (3) `choose_base_tier` then prefers fp8 where the card has it and the gate passed.

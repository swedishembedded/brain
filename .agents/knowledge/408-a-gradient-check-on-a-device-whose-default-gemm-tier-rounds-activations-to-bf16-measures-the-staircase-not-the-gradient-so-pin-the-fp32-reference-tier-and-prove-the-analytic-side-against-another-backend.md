<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 408. A gradient check on a device whose default GEMM tier rounds activations to bf16 measures the staircase, not the gradient: pin the fp32 reference tier, and prove the analytic side against another backend

**Lesson:** A finite-difference check needs a smooth forward, and a device whose default GEMM tier
rounds activations to a narrow format gives a staircase loss that measures step height, not slope.
Pin the full-precision reference tier for gradient checks, and prove the analytic side against
another backend.

## What happened
Question: `qwen35_lora_on_a_bf16_base_analytic_grads_match_finite_differences` failed on CUDA only,
with LoRA-A analytic gradients 49-84 % off the finite differences, and passed on CPU and Vulkan. Was
a CUDA backward wrong?

Bisect (gpu1, tiny model, rank-2 LoRA, bf16 base):

1. `BRAIN_NO_PROVIDER=cuda,library,coopmat,native-f16` and `BRAIN_NO_KERNEL_UPGRADE=1` changed
   nothing, so the generated and native kernel tiers were not involved: the switch that matters is
   not a provider but the GEMM tier (`GemmPrecision::Auto`), which the provider switches do not
   reach.
2. Same weights on CPU and CUDA, one forward and backward, no optimizer: loss 3.4172585 (CPU) vs
   3.417283 (CUDA); every LoRA gradient within 1.5e-3 (f32 base) and 5e-3 (bf16 base) relative L2 of
   the CPU's. The analytic backward is right, to bf16 rounding.
3. After the test's 5 AdamW steps, copy the CUDA weights into a CPU model. Analytic gradients still
   agree to 3e-3, and the CPU's own finite difference along the gradient direction is smooth
   (5.007e-3, 5.078e-3, 5.060e-3 at eps 1e-3, 5e-3, 2e-2 against analytic 5.059e-3). The CUDA finite
   difference of the same weights is 4.7e-2, 7.7e-3, 2.6e-3: at eps 1e-3 the loss moves by 9.4e-5,
   400 ulp of a loss near 2.9, where the true change is 1e-5.

## Why
With a bf16 base the library GEMM tier rounds the activation operand of each product to bf16
(documented in `provider/library.rs`, "Numerics"), so the forward loss is a staircase in every
parameter. A central difference of a staircase measures the step height, not the slope.

CPU and Vulkan run the fp32 kernels, whose loss is smooth. The doc comment already says "gradient
checks and cross-backend parity pin `GemmPrecision::Fp32Reference`"; this one gradient check had
not.

## What to do
`check_qwen35_lora_on` pins `GemmPrecision::Fp32Reference`; it then passes on CUDA and CPU. The
tensor-core tier is held to the reference on training signal by
`crates/qwen35/tests/it/lora_gemm_precision.rs` (loss and gradient cosine 0.999), which is where a
wrong tensor-core backward would show.

Rule: a finite-difference check needs a smooth forward. Before reading an FD mismatch on a new
backend as a backward defect:

1. Repeat the forward at a perturbation ladder and see whether the loss change is monotone.
2. Compare the analytic gradient at identical weights with another backend's analytic gradient.
3. Check which rounding tiers the backend's default turns on.

A mismatch that persists with a smooth forward is a defect; one that disappears when the rounding
tier is pinned is a property of the check.

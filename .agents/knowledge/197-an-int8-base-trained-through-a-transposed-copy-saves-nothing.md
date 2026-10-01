<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 197. An int8 frozen base trained through a transposed copy saves nothing

The plan for an int8 frozen base (the PEFT roadmap's int8 phase) runs the
forward through the packed weight, `y = x·Wᵀ` with the group scale along the
contracted axis `k`, and the input gradient through a second copy, `Wᵀ`
quantised along `out`: `dx = dY·W` contracts over `out`, so the scales of `W`
cannot leave the sum there.

For a diffusion block that is the right trade, because the second copy is
transient or small next to the activations. For a 7B decoder it is not: the two
int8 copies are 2 x 7.6 GB, which is the 15 GB a bf16 base already holds. The
roadmap's claim that int8 "would return about 7 GB to activations" counted one
copy. Nothing is returned, and the bf16 base carries no quantisation error at
all.

The memory comes back with a single weight-only int8 copy, which is what
is built: the forward GEMM, the tiled `dx` GEMM and their plain twins are
the `matmul`, `matmul_reg3`, `matmul_dx` and `matmul_dx_reg` kernels with the
weight binding rewritten by `kernels::template::int8_weight_variant` to four
int8 weights per word, each load decoded as `int8 * scale[wi >> 5]` against
fp32 activations. Both `x·Wᵀ` and `dY·W` read the same packed `W`, so no
transposed copy exists and no activation is quantised: the one approximation
is the weight's rounding. The flat weight index names its group because a
row is a multiple of 32 wide (`wi >> 5` is `row * (k / 32) + col / 32`).

Built over weights the format represents exactly, the model equals the fp32
one (loss and every adapter gradient, `lora_int8_base.rs`); on the real
DeepSeek-R1-Distill-Qwen-1.5B the gradients stay within the bound
`lora_int8_base_real.rs` asserts.

The design that saved nothing, a transposed int8 copy of each weight for the
input gradient, should not be built for a decoder: it holds two int8 copies,
the bytes of a bf16 base.

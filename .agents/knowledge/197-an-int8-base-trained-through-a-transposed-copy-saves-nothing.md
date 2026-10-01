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

The memory only comes back with a single weight-only int8 copy whose GEMMs
decode it inline against fp32 activations, the way the `#w=bf16` variants
decode bf16: both `x·Wᵀ` and `dY·W` read the same packed `W` and apply its
scale per element. That needs a scale binding the `dtype_variant` template does
not have (it rewrites one binding), hence new kernels for the tiled forward
GEMM, the decode GEMV and the tiled `dx` GEMM, and an `Ops` tier to select
them. That is the design to build if a single-card 7B run at long context is
ever required; the transposed-copy design should not be built for a decoder.

A 7B LoRA fine-tune already has two ways to fit: one card at about 3k tokens
with the bf16 base, or two cards at 7.6k tokens (verified), laid out
automatically by free VRAM.

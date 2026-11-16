<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 494. The CPU roof probe ran on the JIT, and a constant-addend FMA costs NEON a register copy per step, so SVE's fmad is the native form

**Lesson:** A rate probe must run on the tier the kernels it grades run on, and its instruction form must match the
hardware's. Check the disassembly of a peak-rate loop before trusting its number: AArch64 `fmla` accumulates into
its destination, so `a = fma(a, c, d)` with a constant addend needs a register copy of `d` per step. SVE `fmad`
(`z = z*m + a`) is that recurrence in one instruction. x86's `vfmadd213ps` already has that form.

## What happened

On the CPU backend `real_work_can_never_beat_the_roof` read `matmul_reg2` at 3206 GFLOP/s against a roof of 591.
The GEMM runs on the native NEON GEMM, while `roof_fma` ran as a Cranelift JIT loop, one invocation per element.
A first native route with eight-accumulator arrays reached only 25.6 GFLOP/s on one Grace core. The disassembly
showed 16 `fmla` and 33 `mov` per step: each FMA needs `d` copied into a fresh destination, and the rolled loop
rotated every accumulator through a spare register (about 17 cycles a step against the 4 the pipes need). Unrolling
by two left one copy per FMA, 55.8 GFLOP/s, half the pipes. SVE `fmad` on the same sixteen chains reached
104 GFLOP/s, the core's full rate (4 pipes x 4 lanes x 2 FLOP x about 3.3 GHz).

## What changed

`crates/backend-cpu/src/fast_ops.rs::roof_fma` is the kernel's native route (matched by source in `FastIdx`). It is
bit-identical to the WGSL arithmetic on every tier (a fused multiply-add rounds once), with sixteen chains per
8-element block. On an SVE host the chains step through an inline-asm `fmad` loop on 128-bit lanes (correct at any
vector length); otherwise through unrolled NEON `vfmaq_f32`; on x86 through AVX2 `_mm256_fmadd_ps`. The roof now
reads 3456-4345 GFLOP/s on 72 cores, and the roofline test runs on the CPU backend instead of skipping it.

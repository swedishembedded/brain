<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 119. A native kernel can be much faster without reassociating anything

**Lesson:** Where a kernel is bound by memory behaviour rather than arithmetic, the fastest rewrite
may cost no numerical agreement at all. Reach for the reassociating one only when the access pattern
is already fixed.

## What happened
The CUDA milestone was planned around a tuned matmul agreeing with the portable reference only to a
tolerance. The reasoning was that a `__shfl` reduction tree reassociates the fp32 sum, so
bit-identity is unachievable.

That reasoning is sound for a reduction tree and it did not apply here.

## Why
The generated tier's problem at this shape is not its arithmetic. Neighbouring threads read the
weight matrix K floats apart, so every lane of a warp touches a different cache line on every
iteration of the reduction.

## What to do
Stage both operands through shared memory and give each thread a register block. This fixes the
ACCESS pattern while leaving the reduction exactly as it was: one accumulator, k ascending.

Keep the assertion at the tolerance the project asserts everywhere else, not the zero that happened
to come out. Pinning the observed value would make the next tuned kernel, which may legitimately
reassociate, look like a regression.

## Evidence
The measured delta over a quarter of a million outputs was zero.

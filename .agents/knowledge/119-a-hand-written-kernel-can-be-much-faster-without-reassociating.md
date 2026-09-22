<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 119. A hand-written kernel can be much faster without reassociating anything

The CUDA milestone was planned around a tuned matmul agreeing with the
portable reference only to a tolerance, on the reasoning that a `__shfl`
reduction tree reassociates the fp32 sum and so bit-identity is unachievable.
That reasoning is sound for a reduction tree and it did not apply here: the
generated tier's problem at this shape is not its arithmetic but that
neighbouring threads read the weight matrix K floats apart, so every lane of
a warp touches a different cache line on every iteration of the reduction.
Staging both operands through shared memory and giving each thread a register
block fixes the ACCESS pattern while leaving the reduction exactly as it was -
one accumulator, k ascending - and the measured delta over a quarter of a
million outputs was zero.

Two things worth carrying forward. Where a kernel is bound by memory
behaviour rather than by arithmetic, the fastest available rewrite may cost
no numerical agreement at all, so reach for the reassociating one only when
the access pattern is already fixed. And the assertion should still be the
tolerance the project asserts everywhere else, not the zero that happened to
come out: pinning the observed value would make the next tuned kernel - which
may legitimately reassociate - look like a regression.

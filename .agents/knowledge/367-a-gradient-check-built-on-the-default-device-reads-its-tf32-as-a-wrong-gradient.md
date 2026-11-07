<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 367. A gradient check built on the default device reads its TF32 as a wrong gradient

**Lesson:** A test whose claim is arithmetic must pin the arithmetic, because the default device
can silently change the numerics underneath it. A test about a device's default must say which
device ran.

## What happened
The EdgeRock gradient checks, green for a milestone, failed after the build environment changed so
that the default device became the CUDA card.

- Layer gradients came back at 70 to 90 percent relative error.
- The numeric side jumped between 0 and 2e-3 where the analytic side held steady.
- The fixed-batch convergence test missed its loss target.

## Why
`Gpu::new` follows the environment. On CUDA the default GEMM tier is TF32 tensor-core products
(`GemmPrecision::Auto`), whose rounding is ~1e-3 relative.

- A central difference of the loss divides that noise by `2 * eps`; the analytic gradient
  differentiates the exact function.
- Under the previous environment the default device was the card through Vulkan, whose GEMMs are
  fp32, so the same tests passed.
- The check had never named the arithmetic it relied on.

## What to do
A check that compares derivatives, or runs that differ only in shape (a batch of two against two
singles), builds its model with `TrainOpts { gemm_precision: GemmPrecision::Fp32Reference, .. }`,
the tier the cross-backend gates are stated against. `edgerock_gradcheck_harness` and
`edgerock_training.rs` do.

A parity test against a reference states its tolerance per tier (`GemmPath`), and the device name
it ran on is printed in its output.

Rule: when a test's claim is arithmetic (derivative, bit equality across shapes), pin the
arithmetic; when it is about a device's default, say which device ran.

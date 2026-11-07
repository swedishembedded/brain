<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 213. A roof probe must issue the instruction it names

**Lesson:** A peak probe must state its instruction explicitly, in a form no backend's arithmetic
contract can split, or every utilisation figure derived from it is inflated.

## What happened
`roof_fma.wgsl`, the compute half of every "% of roof" this engine reports, wrote its chain as
`a = a * c + d`. On Vulkan and wgpu the driver contracts that into one fused multiply-add, and the
probe measured the FMA rate it is named for.

The CUDA generated tier compiles with `--fmad=false` (so that every backend rounds a product and a
sum the way the reference does), and there the same line is a multiply AND an add: two
instructions per two FLOPs, half the issue rate of the silicon.

## Why
It surfaced the moment a native kernel issued explicit `fmaf`. The dense conv kernels for YOLOv8n
training reached 3.7 TFLOP/s at their best shapes against a "measured roof" of 5683 GFLOP/s on a
P40 whose fused peak is about twice that, so a good GEMM would soon have graded itself above 100%
of the device.

If a backend chooses separate rounding for `a * b + c`, as the portable reference requires, an
expression probe silently measures that backend's slower instruction mix.

## What to do
State the instruction explicitly: with `fma(a, c, d)` the same card measured 7996 GFLOP/s (under
contention from another process; the fused peak is the ceiling it approaches when idle).

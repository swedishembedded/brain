<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 6. Timing without `poll_wait()` measures the host

**Lesson:** Timing without `poll_wait()` measures the host. Compute the roof first: a result
above the physical limit means you measured the CPU.

## What happened
`WgpuBackend::submit` with an empty clear list only appends to `pending`; it encodes and queues
nothing. A loop of bare `submit`s therefore times host-side bind-group construction and reports
it as device bandwidth. It produced **377 GB/s on a ~346 GB/s card**.

## What to do
See `.agents/rules/kernels.md` §E.0.

<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 505. The portable axpy reads a third of Hopper's HBM bandwidth through the CUDA backend

**Lesson:** `gpu_core::roof`'s memory roof was the WGSL `axpy` (one word a thread) host-timed around a drained
device. Translated to the CUDA backend it measured 1199-1200 GB/s on a GH200, the same figure on every row of
`brain perf roofline`, while native grid-stride 16-byte kernels on the same card measured a 3686 GB/s read, a
3492 GB/s copy and a 3582 GB/s triad (HBM3, 4000 GB/s published). Every profiler percentage of memory roof on CUDA
divided by a third of the real roof.

## What to do

- `roof::measure` takes the device memory's triad from the native stream kernel (`device_rates::native_triad_gbs`,
  device-timed, fastest of twelve launches) where the backend runs native kernels, and the portable axpy elsewhere.
  The triad's source is in the roof store's key, so the old figure is re-measured, not reused.
- A probe that reaches a backend through translation measures the translation: check it against a native kernel on
  that backend before it becomes a denominator.

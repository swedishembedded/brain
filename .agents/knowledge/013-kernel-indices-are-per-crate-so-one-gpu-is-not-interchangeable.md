<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 13. Kernel indices are per-crate, so one `Gpu` is not interchangeable

**Lesson:** Kernel indices are per-crate, so one `Gpu` is not interchangeable between crates.

## What happened
Building `clip::ClipText` on a `Gpu` constructed from `unet::KERNELS` binds the wrong pipelines.
It surfaced as `Number of bindings in bind group descriptor (4) does not match the bind group
layout (5)`.

That was luck: a wrong index is *silently wrong output*, and it only crashed because those two
kernels happened to differ in arity.

## Why
Every model resolves its kernel ids as positions in the pipeline list ITS crate registered.

## What to do
Use `Gpu::new_like`, which provides a different kernel set on the same device. A pipeline
assembling several crates must use it.

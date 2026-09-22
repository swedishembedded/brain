<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 13. Kernel indices are per-crate, so one `Gpu` is not interchangeable

Every model resolves its kernel ids as positions in the pipeline list ITS crate
registered. Building `clip::ClipText` on a `Gpu` constructed from
`unet::KERNELS` therefore binds the wrong pipelines. `Gpu::new_like` exists for
this - a different kernel set on the same device - and is what a pipeline
assembling several crates must use.

That one surfaced as `Number of bindings in bind group descriptor (4) does not
match the bind group layout (5)`, which was luck: a wrong index is *silently
wrong output*, and only crashed because those two kernels happened to differ in
arity.

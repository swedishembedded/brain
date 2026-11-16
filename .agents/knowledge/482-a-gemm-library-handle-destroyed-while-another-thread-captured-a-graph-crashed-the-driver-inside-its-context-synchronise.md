<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 482. A GEMM library handle destroyed while another thread captured a graph crashed the driver inside its context synchronise

**Lesson:** A vendor library call can synchronise the whole context behind the caller's back: `cublasDestroy` runs
`cuCtxSynchronize`. Anything that synchronises a context while other handles on the device may be capturing graphs
is a drain and takes their capture gates first, as `Context::drain_other_handles` does; a context-wide wait is never
issued bare.

## What happened

`qwen3` `serve_conformance` died with SIGSEGV in about one run in three on a cold kernel cache (CUDA, a GH200). The
crashing thread was dropping the scratch `Ops` of an engine build: `Context`'s drop reached the `cublasDestroy` of its
`Blas`, which called `cuCtxSynchronize` and faulted inside the driver, while the binary's other test threads loaded
modules and one instantiated a graph it had just captured on its own handle. A cold cache made the window wide: every
handle compiled and loaded its kernels, so builds, captures and drops overlapped. Destroying the library handle first,
with its context current, did not change the rate; holding the other handles' capture gates did.

The field order made a second defect beside it: `blas` was the last field of `Context`, so the library handle
outlived the stream it is bound to, against what the comment at its creation said.

## What changed

- `Context::drop` destroys the library handle first, through `Context::destroy_blas`, which holds `OPEN_STREAMS` and
  every other handle's capture gate on the context and makes the context current before `cublasDestroy`.
- Twelve cold-cache runs of `serve_conformance` passed, against two crashes in six before.

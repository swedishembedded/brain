<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 26. A barrier kernel on `backend-cpu` corrupts memory; it does not refuse

**Lesson:** A barrier kernel on `backend-cpu` corrupts memory; it does not refuse. Capability
branches must wrap the recording, and a new kernel test is not done when it is green on the GPU.

## What happened
`DeviceCaps::workgroup_reductions` is false on the CPU JIT because it cannot compile
`workgroupBarrier`. What it does with one is worse than an error.
`crates/gradcheck/tests/it/layernorm2d_kernels.rs` recorded `layernorm_rows` (2 barriers) there
and the process died with

    munmap_chunk(): invalid pointer
    signal: 6, SIGABRT

There was no test name, no kernel name, and no backtrace into the offending dispatch. The
whole-suite runner only said `FAIL: gradcheck suite - CPU backend`; finding which of many test
binaries aborted took longer than the fix.

## Why
- The capability branch must wrap the **recording**, not just the submit; `gpu.step()` on a
  barrier kernel is already too late. Registering it in the kernel table is fine
  (`prelu_kernels.rs` does, and gates the dispatch), which is what made the guard look sufficient
  when it was not.
- A new kernel test is not done when it is green on the GPU. This one shipped measured,
  host-oracled, and red on the other backend, exactly the way lesson #5 says not to. The A/B
  harness is the *easiest* place to make this mistake, because the fast path is the one you are
  excited about and the reference path is the one carrying the barrier.

## Evidence
Running the fused kernel alone on the CPU was not a consolation prize: it matched the host oracle
**exactly** (0.0e0, against 7.2e-7 on the GPU), which is a stronger correctness result than the
GPU run produced.

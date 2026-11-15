<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 465. A cache directory behind a dangling link was answered anyway, so every start recompiled and re-tuned, and NVRTC ran inside the first served rounds

**Lesson:** a cache that is "best effort" must still say when it cannot work. `backend_api::cache_dir()` answered
`~/.cache/brain` although it was a link into a data drive that had been wiped; every store under it (cubins, native
kernel tuning, rooflines, pipeline blobs) failed in silence, and nothing told the operator that every start was a
cold start.

## What happened

A 27B FP8 server on a GH200 behaved as if it had never run before on every start: the native kernel tuner spent its
budget again (and warned that shape classes stayed unmeasured), and the CUDA backend compiled kernels with NVRTC the
first time each shape class appeared. An nsys timeline of `brain perf run serve` (16 streams, 8K prompts, CUDA graph
nodes traced) showed the serving thread inside `nvrtcCompileProgram` (`backend_cuda::nvrtc::compile_image` under
`register_native`) during the largest idle gaps of the first mixed rounds: 38 gaps over 5 ms, together about a
second of an idle card in the first rounds of a run. `~/.cache/brain` held nothing: its target directory no longer
existed, so `create_dir_all` failed at every store.

## What to do

- `cache_dir()` creates the directory on first ask and answers it only when that worked; otherwise it warns once
  (naming the path and the error) and answers `None`, which every caller already treats as "nowhere to persist".
- When a run looks cold on a machine that has run before (tuning warnings, NVRTC in a CPU profile of a serve),
  look at the cache directory first.

## Evidence

`crates/backend-api/tests/cache_dir.rs` (a link into a wiped drive is not answered; a missing, creatable directory is
created). On gpu3 of a GH200 node, `perf run longctx` of the 27B FP8 at 8K context, batch 1: 56 s from start to
exit with an empty cache, 40 s on the next start with the cache filled (no tuning pass); the same run with
`--cache-dir` on a dangling link stayed a cold start.

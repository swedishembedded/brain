<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 462. A device-memory tally must belong to the handle: a process-wide one makes every per-model measurement count its neighbours

**Lesson:** `Backend::memory_usage` is read as "what this model holds". That reading is only true when the tally
behind it is scoped to the handle the model was built on. The WGSL backend kept one per handle; the CUDA backend kept
one static pair (`IN_USE_BYTES`, `PEAK_IN_USE_BYTES`) for the whole process, so on CUDA every figure included every
other handle's buffers and every peak since the process started.

## What happened

The recompute and layer-streaming contracts (`model::saved`, `model::streamed`) and the model tests built on them
compare live or peak bytes on a device the test owns (`gpu_core::testgpu::private`). On CUDA they failed whenever a
sibling test of the same binary allocated, and passed one test at a time: clip pixels read 5,795,708 bytes recomputing
against 960,272 keeping. `toyseq2seq`'s construction-time streaming test failed even with one thread: its peak was
never reset and was the process's peak since start, so it reported 3,517,228 bytes, an earlier test's peak, against
the 308,924 bytes the plain model holds. Alone it passes, so the streamed loader holds less than the plain build, as
the contract requires. `deepseek2ocr` sums three towers' handles and so counted the whole process three times.

## What to do

- A backend's allocation tally belongs to the handle: a handle opened on its own (`new_like` included) starts one,
  a `share`d handle charges its parent's, and a block is uncharged from the tally it was charged to.
- A figure that has to be the model's own is read from the model's own handle; serialising tests is not a substitute
  for a tally that cannot see other handles.
- The same holds for the host-call totals (driver allocations, launches, graph replays): `Backend::call_totals` is
  scoped like `memory_usage`. Read process-wide, `decode_steady_state` saw a sibling test's `cuMemAlloc` calls as its
  own steady-state token's, and every launch-count comparison moved with whatever ran beside it; their per-module
  locks serialised them only against each other.

## Evidence

`crates/backend-cuda/tests/it/backend_contract.rs`
(`memory_usage_counts_this_handle_and_its_shares_only`, red against the process-wide tally), and the
`streamed_construction`, `layer_streamed` and `recompute` tests of `lfm2`, `toyseq2seq`, `clip`, `florence2` and
`sam1` passing at default parallelism. For the call totals: `call_totals_count_this_handle_and_its_shares_only` in
the same file (red against the process-wide totals), and `qwen35`'s `decode_steady_state` passing beside the rest of
its binary.

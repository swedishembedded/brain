<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 519. A helper handle from `share` kept its own scratch arena, so its allocations inside a pooled block forward reached the driver every block

**Lesson:** a scratch scope replays the allocations of the handle it is opened on, and `Gpu::share` gives a handle a
fresh arena of its own. A helper that records into a pass on a shared handle (the `model::ops::Ops` façade) is
outside the pass's scope unless it shares the parent's arena: `Gpu::share_scratch`.

## What happened

After #516, ltxv's quantized block forward builds `Ops::new(gpu.share())` inside the block's scratch scope. Measured on
a GH200 (CUDA, synthetic weights at the real 22B width, 4 layers, 3520 tokens, 128 context, resident session, scratch
pool on, `gpu_core::cuda_call_totals` per warm call, median of 11):

| build | driver allocations / call | zero-fills / call | warm call |
|---|---|---|---|
| before #516 (slot dispatch) | 4 | 111 | 570 ms |
| #516 (`Ops` on `share()`) | 204 | 261 | 449 ms |
| `Ops` on `share_scratch()` | 54 | 111 | 402 ms |

The #516 route was already faster (the selector's kernels, 888 launches against 984), but every block's packed
activations and product temporaries were fresh driver allocations, zero-filled.

## What to do

* A handle that records into a pass of its parent, on the same thread, is `parent.share_scratch()`; a handle another
  thread runs its own scopes on stays `share()` (a lane holds one open scope).
* `crates/gpu-core/tests/it/scratch_arena.rs` pins that the two handles' allocations are one replayed sequence.
* The 50 driver allocations per call the fix leaves over the slot build are not `Ops::new` (which allocates nothing) and
  not allocations a persistent `Ops` would save (one built once per handle keeps its cached temporaries alive, so the
  arena cannot replay them: 204 again). They are open.

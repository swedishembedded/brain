<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 81. A profiler with a silent capacity cliff is a gate that lies by omission - it did not report the wrong number, it reported NO number and let that read as "nothing to see"

`backend-vulkan`'s `flush()` gated its timestamp query pool on `steps.len() <
MAX_TIMED_DISPATCHES` (8192) and simply skipped timing the whole batch above
it - no panic, no log line, `kernel_times()` just came back empty for that
call. The doc comment even framed this as principled ("`kernel_times` stays
honest about what it did and did not see"), but honesty about an omission
still requires SAYING it was omitted; an empty profile and a profile of an
all-idle pass are indistinguishable to whoever reads the table next. This
is exactly the shape that mattered: a 48-layer/128-expert MoE forward
routinely emits tens of thousands of dispatches in one flush, so the one
model-family whose per-kernel attribution `kernels.md` §F.1 most needs
("profile per kernel kind, and publish the table" - the MoE dispatch storm
is `.agents/roadmap/completion-plan.md`'s own Phase 5 #1 item) was
precisely the one this cliff always silently dropped.

There was also no actual Vulkan limit forcing the cliff - `MAX_TIMED_DISPATCHES`
was an implementation choice (query-pool size), not a queried device
capability, so the "correct" fix was never "detect the ceiling and warn",
it was "stop having a ceiling that drops data instead of a ceiling that
bounds a single query pool's SIZE."

**Fix**: `flush` now splits an oversized batch into
`ceil(n / MAX_TIMED_DISPATCHES)` bounded sub-batches (`flush_chunk`), each
its own submit+fence-bounded timestamp bracket, folding every sub-batch's
timestamps into the same per-kernel accumulator - so a batch of any size
gets full per-kernel attribution, at the cost of one extra submit+fence
boundary per `MAX_TIMED_DISPATCHES`-sized chunk (only paid when timing is
on; the untimed path is untouched, one chunk = the whole batch, unchanged
single-submit behavior).

**Rule going forward**: a measurement path that silently produces no data
above some threshold is worse than one that is merely slow or approximate
above it - a missing number reads as "nothing happened" rather than "this
wasn't measured," and nobody re-derives that distinction from a doc comment
while staring at an empty table. When a profiling/gate mechanism has an
internal capacity limit, either lift the limit (chunk, as here) or make the
overflow LOUD (a warning naming the actual count and the cap) - never let
it degrade into silence, which is `.agents/knowledge/` §1's "a gate that never
runs is worse than no gate" restated for measurement instead of correctness.

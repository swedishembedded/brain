<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 544. An adapter bank smaller than the active adapters starves the swapped-out arms without an error

**Lesson:** `--adapter-slots N` swaps an adapter in only when a resident one is idle. When more than N adapters have
requests in flight at once and every resident one stays busy, the others never become resident: their requests wait
with no error, no refusal and no log line, while the resident arms run at full speed. Measured 2026-11-21 (lane EXP,
brain-bcd67e5ad, one GH200, Qwen3-4B `--max-batch 128 --adapter-slots 8 --serve-adapters DIR`): 18 LoRA arms, 6 agent
trials in flight each, 108 in all. After 8 minutes, 6 arms had 15-26 trials graded and the other 12 had 0, each
holding its 6 requests unanswered. The caller's exam was invalid and was discarded.

**Why:** the scheduler stepped over a request whose adapter was still loading (`AdapterAvailability::Loading`) and
admitted the requests behind it, among them new requests for the resident adapters. Each admission pinned a resident
slot again, so with a steady stream of agent turns no slot ever fell idle for the waiting adapter.

**Fix (serving-runtime G13, lane FIX, 2026-11-21):** the generic scheduler (`model::serve`, `next_admissible`) still
admits base requests past a loading adapter, but no longer a later request for a resident adapter: those sequences
drain, a slot falls idle and the waiting adapter takes it, in arrival order. Test:
`crates/model/tests/it/serve_adapters.rs` `a_request_waiting_for_a_slot_is_not_overtaken_by_later_requests_of_resident_adapters`
(red before the fix). With more adapters at work than slots the arms now take turns, each swap costing a drain; give a
server at least as many slots as concurrently active adapters for throughput (rank-16 slots are about 130 MB each),
and still check that every arm progresses before reading a result.

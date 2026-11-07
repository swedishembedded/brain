<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 30. Two of three backends never implemented `stats()`, and a caller papered over it

**Lesson:** Implement the missing `stats()` counters on every backend instead of papering over the
default in a caller; a default trait method is a silent opt-out.

## What happened
`Backend::stats()` defaults to `None` ("this backend does not count device ops"), and the
documented consumer contract is *report null, never zero*. `backend-cpu` and `backend-vulkan` both
took the default.

Nothing noticed, because the single in-tree consumer wrote:

```rust
let before = e.device_stats().map(|s| s.submits).unwrap_or(0);
```

which turns "not counted" into "zero". The test built on it
(`prefill_submits_scale_with_chunks_not_with_token_count`) then compared `0 == 0` in its first two
assertions and passed **vacuously** on the CPU backend. Only its third assertion, `submits > 0`,
ever noticed, and it read as a plain failure rather than as "this backend cannot answer".

## Why
Three things are worth separating, because only the first is obvious:

1. **The fix is the counters, not the caller.** Making the test skip when `stats()` is `None` was
   the first thing tried and it is a workaround: it leaves two backends unable to answer a
   question the trait says they may be asked, and it makes the *next* consumer rediscover this.
   Implementing the counters removes the ambiguity at its source; `None` then becomes an
   assertion failure at the call site rather than a shrug.
2. **`stats()` is per HANDLE, not per device.** The trait says so, and it matters. The counters
   were first put on `backend-cpu`'s `Arc`-shared `CpuShared`, which compiles, passes serially,
   and fails under a multi-threaded test run: every test in the binary shares one pooled device,
   so a neighbour's submits land inside your delta. Per-handle counters make concurrent
   measurement correct by construction; a mutex around the measurement only serialises the test
   against *itself* and does not help. The failure mode of getting this wrong is a **flaky**
   test, which is worse than a wrong one.
3. **A default trait method is a silent opt-out.** `fn stats(&self) -> Option<_> { None }` reads
   as a courtesy to backends that cannot count; in practice it is how two backends went a long
   time without counting while the API looked complete.

## What to do
Implement per-handle counters on every backend. `crates/gpu-core/tests/it/device_stats.rs` now
asserts the contract on whatever backend `BRAIN_DEVICE` selects, so the default cannot creep back.

# continual-state - roadmap

**Status: planned. Nothing below has been started.**

## Goal

Make the recurrent state of the hybrid decoders a first-class, durable object, and make
adapter updates cheap to compare and safe to undo, so that an application can build fast
learning on top of brain without brain deciding what is worth learning.

The Gated-DeltaNet layers of the qwen35 family carry a per-layer delta-rule state
(`model/src/gdn.rs`, 48 layers x 48 value heads x 128 x 128 x 4 B for the 27B: about 150 MiB
per sequence plus a small convolution history, constant in context length). Today that state
lives in a device buffer for the life of one sequence and is dropped on release
(`qwen35/src/serve.rs`, `release_table`). The only snapshot is the in-process one used to
undo a rejected speculative tail (`qwen35/src/model.rs`, `GdnSnapshot`). qwen35 has no prefix
cache at all (`serve.rs` module doc, "Deliberately deferred"). qwen3 has a prefix cache keyed
by adapter namespace, but for KV only.

What this is NOT: no TTT, Titans or HOPE-style trainable inner-loop memory. A persisted
state is a cache of the tokens that produced it. It is not a generalization of them. The
research track for an explicit fast-weight module is out of scope here; its prerequisites
are in the Research track below.

Brain names no agent runtime and no orchestrator. Callers decide which sessions' state is
kept and for how long; brain provides the mechanism and the guarantees.

## Left to do

- [ ] CL-B1 (M) Host-serializable recurrent state with an exact round trip.
  - A `RecurrentSlot` can be written to bytes and restored into a fresh slot. The header
    carries a format version, the config digest, the weights digest, the ADAPTER digest
    (state computed under one adapter is invalid under another), the token count, and a hash
    of the consumed token prefix. A checksum covers the payload. Any mismatch is a typed
    error, never a silent restore.
  - Test first, on the tiny qwen35 config on the CPU backend: prefill(A), save, restore into
    a fresh slot, continue with B, and compare logits with prefill(A+B) at the backend's
    parity tolerance. Also: restore under a different adapter digest is refused; a flipped
    payload byte is refused; restoring twice gives the same continuation.
  - Acceptance: the test above is red on HEAD (no API) and green after. `make test
    test/parity check`.
- [ ] CL-B2 (L) Hybrid-aware prefix cache for qwen35 (after CL-B1).
  - A cache entry for a hybrid model holds the KV blocks of the prefix and a recurrent-state
    snapshot at the same boundary, keyed like qwen3's cache by adapter namespace plus token
    prefix. Entries are refcounted and evictable under the memory authority's budget.
  - A prefix hit must produce exactly the logits of a cold prefill. Throughput claims need a
    measurement (CL-B4), not the design.
  - Acceptance: a conformance test that a warm and a cold request give equal logits, a
    namespace test (a base request never adopts an adapter's entry), an eviction test under
    a tiny budget. `make test test/parity test/e2e/api-conformance`.
- [ ] CL-B3 (M) Paired evaluation of an incumbent and a candidate adapter in one process.
  - With the adapter bank (qwen3) the base, the incumbent and the candidate are resident at
    once. Expose an SDK call that runs the same prompt set against two adapter ids with the
    same seeds and returns per-prompt outputs and token logprobs for both, so a caller can
    run a paired test without a model reload between the two.
  - Acceptance: a bank test in which two adapters with known different behaviour are
    compared on a fixed prompt set; outputs equal the single-adapter servers' outputs at
    equal batch composition (batched kernels are not bit-stable across compositions, so the
    comparison is by logits). `make test test/e2e/api-conformance`.
- [ ] CL-B4 (S) Measure what the design depends on, on the GH200, and record it.
  - LoRA step time on qwen3 8B and qwen35 27B at stated sequence lengths (the existing
    qualification gate for one SFT step), prefill throughput of qwen35 for a stated prompt
    length, and the time to save and restore a state (CL-B1) from pinned host memory and
    from disk.
  - Acceptance: numbers recorded in a knowledge entry with the command that produced them
    and the sharing state of the device. These are the numbers the latency budget of any
    caller is derived from; until they exist, every such budget is an estimate.
- [ ] CL-B5 (M) Byte-equal rollback on a live adapter swap.
  - The follow path (`--adapter-manifest`, `--watch-adapters`) already offers a new release
    beside the old and then withdraws the old one. Add the converse as a tested guarantee:
    withdrawing the new release (rollback) leaves the responses to a recorded probe set
    byte-identical to those before the swap, compared at equal batch composition.
  - Shared with the continual reader's V17 check; one implementation.
  - Acceptance: `a_rolled_back_swap_leaves_the_live_response_byte_identical`-style test in
    the catalog or cli adapter tests. `make test check`.

## Research track (not scheduled)

An explicit test-time-training or neural-memory module (Titans, TTT layers, HOPE) is a
separate effort with prerequisites that are not met:

- A measured benefit of persisted state (CL-B1 then an experiment by the caller that carries
  state across sessions and measures recall after the text and KV are gone).
- A published recipe with released weights for an architecture brain supports, or a
  meta-training budget. As of 2026-10-16 no official weights were found for Titans or HOPE;
  the one recipe for released checkpoints needs multi-GPU continued pretraining.
- Kernels for an inner-loop update (the linear case is the existing GDN chunk kernel
  family) and a backward through that loop.

## Invariants

- State, adapters and caches fail closed on any digest mismatch.
- A restored state is never served under a different adapter or weights than produced it.
- Nothing here edits the weights of a running model; adapter changes go through the existing
  release-following path.

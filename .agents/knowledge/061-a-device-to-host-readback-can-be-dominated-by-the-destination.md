<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 61. A device-to-host readback can be dominated by the DESTINATION allocation, and a correctness gate cannot see a performance mechanism at all

Two separate findings from one change; they arrived together and each is the
kind that gets re-discovered.

**The readback is not necessarily bus-bound.** `Gpu::read` on a P40 measured
0.82 GB/s for a 206 MiB activation against 5.87 GB/s for the same payload in
the other direction, which reads as a PCIe asymmetry and is not one. Split
into phases, the bus transfer (`copy_buffer_to_buffer` + submit + poll) was
35 ms and the host `get_mapped_range() -> to_vec()` was 150 ms - the *same*
copy into a pre-faulted sink took 22 ms. The cost was the destination `Vec`'s
first-touch page faults: brand-new anonymous pages, zeroed and faulted as the
memcpy writes them. Reading the mapped range was never slow.

The general shape, and why it matters beyond one backend:

* **A transfer path has at least three phases and only one of them is the
  bus.** Allocate/pin, transfer, and copy-out. Quoting an end-to-end GB/s
  attributes all three to whichever one the reader assumes. Split it before
  building anything on the number - the split here was four `Instant`s behind
  a temporary env var and took minutes.
* **Allocation is the recurring culprit in this workspace, in BOTH
  directions.** Upload staging on non-ReBAR cards was the same class
  (`crates/gpu-core/tests/vram_overhead.rs`); this is the read direction, and
  the destination half of it is still open because `read`'s contract is to
  return a fresh `Vec`.
* **A rate that is wildly asymmetric between two directions of the same link
  is a claim about software, not about hardware.** PCIe is symmetric to within
  a few percent.

**And a correctness gate cannot see a performance mechanism.** Reusing the
staging buffer instead of allocating one per call is a pure performance
change, so every correctness test - the ones that pin "a read returns its own
bytes and not the previous read's tail" - passes just as well for a `read`
that allocates every time. Mutating `retain_read_staging` into a no-op, i.e.
silently reverting the whole mechanism, left all four correctness tests
green. The mechanism needed its own OBSERVABLE
(`WgpuBackend::read_staging_allocations`, asserted as "eight reads of one
shape allocate exactly one buffer") before a gate could exist at all.

So: when a change is "the same answer, computed with less work", ask what in
the process state proves the work was skipped, and expose that. A wall clock
is not a gate - nobody watches it, and it is the first thing a noisy machine
takes away. `crates/backend-wgpu/tests/read_staging_reuse.rs`, and the numbers
in `.agents/roadmap/ltxv.md` phase 36.

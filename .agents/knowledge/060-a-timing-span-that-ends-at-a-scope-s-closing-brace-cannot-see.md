<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 60. A timing span that ends at a scope's closing brace cannot see the DESTRUCTION its own scope causes, and that is where half the cost was

The LTX-2.5 block forward was instrumented in four buckets - host modulation,
graph record + upload, submit + wait, readback - and each was reported on its
own. Nothing ever compared their SUM against the wall clock of the thing that
contains them, so nobody could see that they did not add up. They did not:
every timing span closed before the function returned, and the last thing
the function does is drop ~70 device buffers, which on a discrete card is ~70
driver frees. That was 10.5 s of a 30 s host half, in a forward whose recorded
"next thing to fix" was the ALLOCATION of the same buffers - measured at
roughly the same size, and treated as if it were the whole cost.

The general shape:

* **Instrument the CALLER of the scope, not just spans inside it.** A bracket
  around `f()` catches what `f`'s own internal spans structurally cannot: its
  drops, its returns, its allocator work. Here the fix was one probe around the
  whole function and one around the loop that calls it, and the residual it
  exposed - 1.9 s per 4 blocks that belonged to no bucket - was the entire
  finding.
* **A stage table that "roughly adds up" is not attribution.** Sum the spans
  and subtract from the enclosing wall clock EVERY time, print the residual as
  its own row, and treat a nonzero residual as a measurement to be made rather
  than as noise. `.agents/roadmap/ltxv.md` phase 33's table has an explicit
  `unaccounted` row for this reason; it reads 0.00 s because the probes were
  added until it did.
* **`free` is a cost, not a cleanup.** Rust makes deallocation invisible at the
  call site, so a profile built from hand-placed spans will systematically
  under-count it. This is the second time device-buffer lifetime has cost this
  workspace real wall clock (the first was staging buffers on non-ReBAR cards,
  `crates/gpu-core/tests/vram_overhead.rs`).

The corroborating check is cheap and was decisive: an external sampling
profiler (`perf record --call-graph dwarf`, `--delay` past the cold phase)
showed `gpu_allocator::vulkan::Allocator::free` and `MemoryBlock::new` at
comparable shares, and both left the profile entirely once the buffers were
pooled. When an in-process attribution says something surprising, one
`perf` run either confirms it from outside or kills it.

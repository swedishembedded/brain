<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 204. A cache that holds a buffer strongly makes the device handle an owner of the model

The requirement is that brain never leaks CUDA memory. `cuMemGetInfo` cannot
prove that on a shared card (anyone else's allocation moves it), and
compute-sanitizer's leak check only sees what is still allocated at process
exit - by which point the handle that kept it resident is long gone. Neither
sees a *retention*: memory that is reachable, freed eventually, and unusable in
between.

`backend_cuda::live_resources()` counts outstanding device and pinned
allocations, events, streams, graphs, graph execs, modules and primary-context
retains. Each is incremented when the driver confirms a create and decremented
only when it confirms the free, so a refused free stays visible. `tests/leaks.rs`
runs create/drop loops of every owner and asserts the counters return EXACTLY,
then `cuMemGetInfo` within a tolerance (which catches an owner that forgot to
register).

It found three retentions, all of the same shape - a per-handle cache holding a
strong reference to something the model owns:

- A captured graph held an `Arc` to every buffer its nodes name. A model that
  dropped its buffers while the device handle lived on (a handle serves the
  next model) left every weight resident until some later submission evicted
  the graph: 301 MB in the test, ~28 GiB for a real model.
- The per-step uniform table kept a device block and a fence event for steps
  whose buffers were gone: 62 leaked blocks and 62 events after 60 buffer
  generations. It was only swept at 4096 entries.
- The eager path took one `cuMemAllocHost` block per step (32 steps made 32
  allocations; ~10 KiB of driver bookkeeping resident each) from a pool of up
  to 8192.

And one lifetime defect: `Graph`, `GraphExec` and `Module` did not hold the
primary context, so dropping one after the last handle hung the process inside
the driver (the test timed out; it did not fail).

Fixes: graphs hold `Weak` (the allocator epoch already discards a graph captured
before any free, which is what actually prevents a stale replay); `poll_wait`
releases graphs of an older epoch and uniform blocks of dead buffers; staging is
a bounded slab pool (4 MiB) reset at drain; every driver object holds the
context.

Rule: a cache keyed by identity must hold the identity weakly and rely on an
explicit invalidation signal, never on holding the object alive. And a leak
gate must assert on exact counts at the point the owner's *model* is gone while
its *handle* lives on - dropping the handle too hides this whole class.

<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 104. A `poll_wait()` inside the loop body reclaims nothing, because the object holding the buffers is still alive when it runs

Dropping the last host handle to a device buffer does not return its memory.
wgpu may only reclaim a buffer once the GPU is known to be done with every
submission that could reference it, which it learns from a completed
`poll_wait`. Until then the bytes are *pending*, and a loop that streams one
layer at a time and never reclaims accumulates the whole model - which is
what `WgpuBackend::track`'s ceiling assertion exists to catch instead of an
opaque "wgpu error: Out of Memory" hours later.

The trap is not "forgetting to poll". It is that the obvious poll is in the
wrong place and looks right:

```rust
for l in 0..n {
    let layer = build_layer(&gpu, l);   // this layer's weights, uploaded
    x = layer.forward(&x);
    gpu.poll_wait();                    // WRONG - `layer` is still alive
}
```

At that poll, `layer` is a live binding; nothing of its is pending, so the
poll waits on a device that has nothing to hand back. `layer` drops
immediately afterwards, and its bytes wait for the NEXT poll - which arrives
only after the next iteration has already allocated. The freed bytes land one
iteration late, forever, and the ceiling still trips. Order is the entire
content of the fix: **drop, then poll**. That is not a fact about this loop;
it is a fact about every loop of this shape, and the same wrong version was
written first, independently, in two different crates on one day.

Two consequences worth carrying:

**A readback hides the problem exactly where it does not fix it.** `read`
polls internally, so a block whose `forward` ends in a readback clears the
counter every iteration - while it is alive. The accumulation happens in the
window between the drop and the next allocation, and no amount of reading
inside the body covers that window. This is why the bug survived so long in
paths that looked "obviously drained".

**"It has tests and they pass" proves nothing here.** Every model crate's
unit tests use tiny synthetic configs whose entire weight set is a few
megabytes, so the ceiling is unreachable. Both real incidents surfaced only
under real 12B/22B checkpoints. A test of this mechanism has to be a test of
the MECHANISM (`crates/gpu-core/tests/transient_reclaim.rs`: observe
`pending_reclaim_bytes` directly, and lower the ceiling with
`BRAIN_GPU_RECLAIM_CEILING` so a megabyte loop can cross it), never a bigger
model test.

**The object holding the memory is not always a "layer".** Two of the real
instances found by sweeping for this had no per-layer object at all:

* A `Gpu` HANDLE owns device memory of its own - its scratch replay arena
  above all. `DitSession::device_for_call` handed a fresh `Gpu::share` to
  every forward, and dropping that handle at the end of the call abandoned a
  measured **3.13 GB per forward** with nothing left to reclaim it, because
  the session's own handle - the only other one on that device - was never
  asked to. A handle that IS the device (the transient session shape) has the
  opposite property: dropping it tears the device down and frees everything,
  so it needs no reclaim at all. `ltxv::devres::CallDevice` now makes which
  is which a type distinction rather than a thing to remember.
* Per-iteration weight uploads written as a plain function
  (`upload_det_block(gpu, ..) -> DetBlockWeights`) rather than a
  `Type::on(..)` constructor. Grepping for constructor-shaped call sites
  misses these entirely; `crates/ltxv/src/na_decoder.rs` had two, and the
  ceiling assertion in a real-weight test is what found them.

**The counter over-counts a record-then-submit graph, and that is not a
bug to fix in the counter.** A loop that RECORDS dispatches into a step list
and submits after the loop drops its intermediate host handles per iteration
while the recorded bind groups still reference the buffers - so
`pending_reclaim_bytes` counts them as abandoned when they are deliberately
alive, and a poll placed in such a loop could not reclaim them anyway.
`EmbeddingsConnector::forward` is one (1.68 GB across 16 blocks). At the real
ceiling it never fires; it only shows up if the ceiling is lowered for a
test, which is worth knowing before reading such a report as a leak.

The structural fix is `gpu_core::Transient` - an RAII guard that drops its
payload and *then* polls - plus `gpu_core::reclaiming(gpu, || ..)` for a body
that allocates buffers with no single owner. The load-bearing part is not
that the helpers exist but that the per-layer CONSTRUCTORS return the guard
(`LtxBlock::on`, `LtxAvBlock::on`, `LtxBlockQ::on`, `LtxAvBlockQ::on`,
`EmbeddingsConnector::on`, `Gemma4Layer::on`, `WanBlock::on`): there is no
way to obtain a bare per-iteration block, so the loop above cannot be written
at all, whether or not its author ever read this entry. Where a constructor is
genuinely dual-use, residency is what gets the explicitly named variant
(`LtxBlockQ::resident_on_cached`), so the safe shape stays the default and the
kept-alive shape is the one that has to say so.

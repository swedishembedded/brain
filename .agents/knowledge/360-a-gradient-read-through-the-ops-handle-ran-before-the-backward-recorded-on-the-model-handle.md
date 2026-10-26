<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 360. A gradient read through the `Ops` handle ran before the backward recorded on the model handle

Symptom: a layer-streamed full fine-tune of qwen35 ended at a different loss
than the resident one, but not every time. The same configuration matched bit
for bit on one run and differed on the next, on the default wgpu backend. The
gradients left in the buffers after the step were identical in both builds; the
update had read different values.

Cause: the model records its backward on its own `Gpu` handle, and
`model::Ops` is built on `gpu.share()`, a second handle on the same device.
On wgpu every handle keeps its own list of recorded dispatches and sends them to
the queue only when it flushes (a `read`, `poll_wait`, `flush`, or a submission
with clears). `ResidentFullFt` reads gradients, clears gradient lanes and
rewrites weights through the `Ops` handle. A copy or clear issued there reached
the queue ahead of backward work the model handle had recorded but not yet sent,
so the update read a gradient that was only partly accumulated, and a lane
cleared on the `Ops` handle could be zeroed after the backward had written it.
The CUDA backend orders handles with per-buffer fences, but only for work that
has been issued: held work records no fence either.

The race was already present in the layer-streamed gradient-norm read. It
surfaced once the embedding and head moved into the resident state, because
their gradients are read after the whole backward, when the most work is
pending on the model handle.

Fix: the ordering is a contract of `ResidentFullFt` (its doc). The model sends
its handle's recorded work to the device (`Gpu::flush`) before every call that
touches device data: the generic driver `resident_ft::adamw_step` does it
through `ResidentBackward::flush_backward` before each read, update and layer
event, and a model calling in directly (loading a layer, reading a master,
committing) flushes first. `ResidentFullFt` flushes the `Ops` handle after the
clears it issues, so the backward that accumulates next comes after them.

Rule: two handles on one device are two queues of recorded work. Data produced
on one handle and consumed on another needs the producer flushed before the
consumer's submission, whichever backend runs it. A single-handle test (the
model-level `resident_ft` specs) cannot see this; the test that found it builds
a real model and repeats the comparison.

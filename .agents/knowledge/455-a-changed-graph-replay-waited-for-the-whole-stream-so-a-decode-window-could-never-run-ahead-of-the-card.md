<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 455. A changed graph replay waited for the whole stream, so a decode window could never run ahead of the card

**Lesson:** A hazard that belongs to one resource must be waited out on that resource. The CUDA graph
cache guarded each captured graph's parameter staging with a single "a replay is in flight" flag and
`cuStreamSynchronize`, so rewriting one small graph's staging waited for every graph queued behind it.

## What happened

On a Tesla P40 serving Qwen3-0.6B, nsys showed each sub-step of the on-device decode window (feed,
advance, layer stack, head, with no readback in between) block in `cuStreamSynchronize` inside the replay
of the two-node feed/advance graph, whose parameters name the sub-step. The host could only resolve and
launch the next layer stack once the card was idle, so the card idled about 0.9 ms of every 17 ms
sub-step. gdb stack sampling of the serving thread put 113 of 120 samples in that `sync`.

## Why

- The staging block a replay rewrites is private to its graph, and only that graph's copy node reads it.
  The work queued after that graph's last launch (the layer stack, the head) never reads it.
- The flag's own comment assumed a decoder reads its logits between submissions, which drains the
  device anyway. The decode window was built precisely to stop doing that, and nothing re-examined the
  assumption.

## Fix

Each captured graph records an event after every launch (`LiveGraph::launched`); a replay that changes
the graph waits on that event only, and counts a staging wait only when the event had not completed.
Discarding or recycling a graph still settles the stream. Gate:
`cuda_graphs::a_changed_replay_does_not_wait_for_another_graph_in_flight` (deterministic: the device is
drained after `A`'s launch, then `A` is replayed with new parameters behind an in-flight `B`). Numbers:
`.agents/roadmap/serving-runtime.md`, "Qwen3 serving on a Tesla P40".

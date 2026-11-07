<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 431. A server slowed down with age because long prefill scratch outgrew the 4 GiB block cache and went back to the driver every round

**Lesson:** A server can slow down with age when a transient working set outgrows what the allocator
is allowed to keep. Profile a cell after the earlier cells on one server, and let the allocator
retain what the planner already reserves for that working set.

## What happened

Question (2026-11-02): the goal matrix on one server gave 71 tok/s at 32K x 16 where a fresh server
gave 84. Why does a server get slower with the traffic it has served?

- Same binary, same card. 32K x 16 was 84.5 tok/s on a fresh server and 72.8 tok/s after an 8K x 64
  cell. Before the retained-state-row change it was 74.8 after the same history, so that change was
  not the cause.
- nsys over the 32K x 16 cell: the kernels ran at the same per-launch times, but GPU idle gaps
  longer than 1 ms grew from 5.5 s to 17.7 s of a 95 s window. Memcpys were under 0.1 s in both
  runs.

## Why

- CPU sampling inside the gaps: `cuMemAllocFromPoolAsync` under `Gpu::storage_with` in the prefill
  layer stack (`layer_gdn_fwd_pre`, `mlp_fwd_core`, `rms_quant_front`), and the round's `read_into`
  sync waiting behind it.
- The block cache and the pool's release threshold default to a sixteenth of the card, at most 4
  GiB. A 32K round's per-layer scratch is larger, so after a history that moved the free memory, the
  pool returned it at every synchronisation and each layer mapped it again.

## What to do

- `--block-cache-mb 16384` restored a fresh server's speed: 32K x 16 77.9 -> 85.1 tok/s and 8K x 64
  339 -> 360. The fix needs no flag. The serving instance retains freed blocks up to
  `ServingShape::stage_workspace`, the memory the placement already keeps free for a prefill round
  and a decode step. With it: 85.0 and 372.9.

### Rules

- When the same cell is slower late in a matrix than on a fresh server, profile it after the earlier
  cells on one server. A fresh-server profile hides costs that come from the server's history.
- A transient working set the planner reserves memory for should also be what the allocator may
  keep. A fixed fraction of the card below it turns every round into driver calls.

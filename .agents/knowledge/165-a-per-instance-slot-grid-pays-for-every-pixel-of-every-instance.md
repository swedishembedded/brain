<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# 165. A per-instance slot grid pays for every pixel of every instance

**Lesson:** Before optimizing a stage, multiply out the bytes it moves per frame. A layout sized
items x pixels-per-item x channels is a bandwidth bill even when every kernel on it is fast.

## What happened

The ray renderer's backward wrote each (tile, gaussian) instance's 17 gradient channels for all 256
pixels of its tile into a slot grid, and a second kernel read the grid back to reduce each instance.

Profiled on a 500k-gaussian scene at 816x612 it took about 480 ms per view, 92% of the fit. The
suspect was band replays of crowded tiles.

## Why

The cause was plain traffic: two million instances x 17 channels x 256 pixels x 4 bytes is ~35 GB
written and read per view, for a result of 68 bytes per instance. Most of those pixel slots were
zeros, because a gaussian covers a small part of the tile it was binned into.

## What to do

**Rule:** before optimizing a stage, multiply out the bytes it moves per frame. A layout whose size
is (items x pixels-per-item x channels) is a bandwidth bill even when every kernel on it is
individually fast; keep the per-pixel partials on chip and write only the reduction.

A workgroup per tile that walks the list in lockstep and reduces each instance in workgroup memory
(`splat_ray_bwd_tile.wgsl`) writes only the 68 bytes. The slot kernel stays as the per-pixel
reference and the CPU path.

## Evidence

45.6 ms per view with the tile workgroup, against the slot grid time above.

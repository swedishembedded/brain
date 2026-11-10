// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Advance a position counter held in a buffer: pos[0] += 1 (the last dispatch of a replayable decoder step)
// @how   one thread
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Params: none used (a dummy word). Dispatch: 1 thread.

struct Params { unused: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> pos: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x == 0u && gid.y == 0u) { pos[0] = pos[0] + 1u; }
}

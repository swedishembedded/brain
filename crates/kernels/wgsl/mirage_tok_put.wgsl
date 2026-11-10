// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Token history I/O of the device decoder loop: hist[b, pos + 1] = u32(picked[b]) (the greedy token chosen at pos)
// @how   one thread per element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// picked: [batch] token ids as f32 (argmax_row output). Params: batch, maxt, pos. Dispatch: batch threads.

struct Params { batch: u32, maxt: u32, pos: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       picked: array<f32>;
@group(0) @binding(2) var<storage, read_write> hist: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.batch) { return; }
    hist[idx * p.maxt + p.pos + 1u] = u32(picked[idx]);
}

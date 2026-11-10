// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training column reduction: dst[c] += sum over rows of g[r, c] (times o[r, c] when used)
// @how   one thread per column, looping over the rows (bias, channel-scale and scalar gradients)
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The gradient of a per-channel bias (use_o 0) or scale (use_o 1, o is the scaled input) over `rows` rows of
// width `period`. A scalar gradient is period 1. One owner per output, so the sum is deterministic; the loop over
// rows is untuned (a two-pass tree is the follow-up for tall inputs).
// Params: rows, period, use_o. Dispatch: period threads.

struct Params { rows: u32, period: u32, use_o: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       g: array<f32>;
@group(0) @binding(2) var<storage, read>       o: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.period) { return; }
    var s = 0.0;
    for (var r = 0u; r < p.rows; r = r + 1u) {
        let i = r * p.period + idx;
        if (p.use_o != 0u) {
            s = s + g[i] * o[i];
        } else {
            s = s + g[i];
        }
    }
    dst[idx] = dst[idx] + s;
}

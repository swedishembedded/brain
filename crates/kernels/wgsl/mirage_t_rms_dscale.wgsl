// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training RMS-norm scale gradient: ds[j] += sum over rows of g[r, j] * x[r, j] * inv[r]
// @how   one thread per channel, looping over the rows
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Params: rows, d. Dispatch: d threads.

struct Params { rows: u32, d: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       g: array<f32>;
@group(0) @binding(3) var<storage, read>       stats: array<f32>;
@group(0) @binding(4) var<storage, read_write> ds: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.d) { return; }
    var s = 0.0;
    for (var r = 0u; r < p.rows; r = r + 1u) {
        s = s + g[r * p.d + idx] * x[r * p.d + idx] * stats[r * 2u];
    }
    ds[idx] = ds[idx] + s;
}

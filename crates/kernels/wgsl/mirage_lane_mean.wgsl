// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Mean over the lanes of a hyper-connection stack: out[b, c] = mean_j lanes[b, j, c]
// @how   one thread per output element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Params: rows, n, d. Dispatch: rows*d threads.

struct Params { rows: u32, n: u32, d: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       lanes: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.rows * p.d) { return; }
    let c = idx % p.d;
    let b = idx / p.d;
    var acc = 0.0;
    for (var j = 0u; j < p.n; j = j + 1u) { acc = acc + lanes[(b * p.n + j) * p.d + c]; }
    out[idx] = acc / f32(p.n);
}

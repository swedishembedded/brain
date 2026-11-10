// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Hyper-connection write-back: next[b, i, c] = sum_j h_res[b, i, j] * lanes[b, j, c] + h_post[b, i] * (y[b, c] - u[b, c])
// @how   one thread per output element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Params: rows, n, d. maps stride is 2n + n*n. Dispatch: rows*n*d threads.

struct Params { rows: u32, n: u32, d: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       lanes: array<f32>;
@group(0) @binding(2) var<storage, read>       y: array<f32>;
@group(0) @binding(3) var<storage, read>       u: array<f32>;
@group(0) @binding(4) var<storage, read>       maps: array<f32>;
@group(0) @binding(5) var<storage, read_write> next: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.rows * p.n * p.d) { return; }
    let c = idx % p.d;
    let i = (idx / p.d) % p.n;
    let b = idx / (p.d * p.n);
    let stride = 2u * p.n + p.n * p.n;
    var acc = maps[b * stride + p.n + i] * (y[b * p.d + c] - u[b * p.d + c]);
    for (var j = 0u; j < p.n; j = j + 1u) {
        acc = acc + maps[b * stride + 2u * p.n + i * p.n + j] * lanes[(b * p.n + j) * p.d + c];
    }
    next[idx] = acc;
}

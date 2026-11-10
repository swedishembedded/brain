// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training token cross-entropy backward: dx[r, j] += w[r] g (softmax(x[r])[j] - [j == tgt[r]])
// @how   one thread per logit; accumulates
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// lse from mirage_t_ce_rows, g the scalar loss gradient. Params: total, v. Dispatch: total threads.

struct Params { total: u32, v: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       tgt: array<u32>;
@group(0) @binding(3) var<storage, read>       w: array<f32>;
@group(0) @binding(4) var<storage, read>       lse: array<f32>;
@group(0) @binding(5) var<storage, read>       g: array<f32>;
@group(0) @binding(6) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let r = idx / p.v;
    let j = idx % p.v;
    if (w[r] == 0.0) { return; }
    var d = exp(x[idx] - lse[r]);
    if (tgt[r] == j) { d = d - 1.0; }
    dx[idx] = dx[idx] + w[r] * g[0] * d;
}

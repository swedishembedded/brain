// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Hadamard-MLP middle stage: z = silu(d2 * (1 + softmax(logits) . cond_u) * z + b2) with a rank-r conditioning per row
// @how   one thread per element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// z: [rows, n] (in place), logits: [rows, rank] (softmaxed here, per thread: the rank is small), cond_u: [rank, n], d2, b2: [n].
// Params: rows, n, rank. Dispatch: rows*n threads.

struct Params { rows: u32, n: u32, rank: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> z: array<f32>;
@group(0) @binding(2) var<storage, read>       logits: array<f32>;
@group(0) @binding(3) var<storage, read>       cond_u: array<f32>;
@group(0) @binding(4) var<storage, read>       d2: array<f32>;
@group(0) @binding(5) var<storage, read>       b2: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.rows * p.n) { return; }
    let c = idx % p.n;
    let r = idx / p.n;
    var mx = -1.0e30;
    for (var k = 0u; k < p.rank; k = k + 1u) { mx = max(mx, logits[r * p.rank + k]); }
    var den = 0.0;
    var mix = 0.0;
    for (var k = 0u; k < p.rank; k = k + 1u) {
        let e = exp(logits[r * p.rank + k] - mx);
        den = den + e;
        mix = mix + e * cond_u[k * p.n + c];
    }
    let cond = 1.0 + mix / den;
    let a = d2[c] * cond * z[idx] + b2[c];
    z[idx] = a / (1.0 + exp(-a));
}

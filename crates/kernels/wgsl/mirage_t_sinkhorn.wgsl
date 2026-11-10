// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training Sinkhorn forward: doubly stochastic exp(normalised logits) of each n x n block
// @how   one thread per block, function-scope array of 36
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Sinkhorn normalisation in log space of one n x n block of logits per thread (n <= 6): `iters` rounds of row then
// column log-sum-exp normalisation, then exp. The arrays are fixed-size function-scope arrays (the blocks are tiny:
// the real width is 4 lanes).
// Params: blocks, n, iters. Dispatch: blocks threads.

struct Params { blocks: u32, n: u32, iters: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>  x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.blocks) { return; }
    var l: array<f32, 36>;
    let nn = p.n * p.n;
    let base = idx * nn;
    for (var i = 0u; i < nn; i = i + 1u) { l[i] = x[base + i]; }
    for (var it = 0u; it < p.iters; it = it + 1u) {
        for (var i = 0u; i < p.n; i = i + 1u) {
            var mx = l[i * p.n];
            for (var j = 1u; j < p.n; j = j + 1u) { if (l[i * p.n + j] > mx) { mx = l[i * p.n + j]; } }
            var s = 0.0;
            for (var j = 0u; j < p.n; j = j + 1u) { s = s + exp(l[i * p.n + j] - mx); }
            let lse = mx + log(s);
            for (var j = 0u; j < p.n; j = j + 1u) { l[i * p.n + j] = l[i * p.n + j] - lse; }
        }
        for (var j = 0u; j < p.n; j = j + 1u) {
            var mx = l[j];
            for (var i = 1u; i < p.n; i = i + 1u) { if (l[i * p.n + j] > mx) { mx = l[i * p.n + j]; } }
            var s = 0.0;
            for (var i = 0u; i < p.n; i = i + 1u) { s = s + exp(l[i * p.n + j] - mx); }
            let lse = mx + log(s);
            for (var i = 0u; i < p.n; i = i + 1u) { l[i * p.n + j] = l[i * p.n + j] - lse; }
        }
    }
    for (var i = 0u; i < nn; i = i + 1u) { y[base + i] = exp(l[i]); }
}

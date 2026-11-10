// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training Sinkhorn backward: replay the rounds into scratch, then back-propagate each half step
// @how   one thread per block, function-scope arrays of 36, trajectory in a global scratch buffer
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
// traj holds the state after every half step: [blocks, 2 iters, n n]. The gradient of a half step that divides by a
// sum (rows or columns) is g - exp(out) * sum(g) along its axis.
// Params: blocks, n, iters. Dispatch: blocks threads.

struct Params { blocks: u32, n: u32, iters: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>  x: array<f32>;
@group(0) @binding(2) var<storage, read>  y: array<f32>;
@group(0) @binding(3) var<storage, read>  gy: array<f32>;
@group(0) @binding(4) var<storage, read_write> traj: array<f32>;
@group(0) @binding(5) var<storage, read_write> dl: array<f32>;

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
        for (var i = 0u; i < nn; i = i + 1u) { traj[((idx * 2u * p.iters) + 2u * it) * nn + i] = l[i]; }
        for (var j = 0u; j < p.n; j = j + 1u) {
            var mx = l[j];
            for (var i = 1u; i < p.n; i = i + 1u) { if (l[i * p.n + j] > mx) { mx = l[i * p.n + j]; } }
            var s = 0.0;
            for (var i = 0u; i < p.n; i = i + 1u) { s = s + exp(l[i * p.n + j] - mx); }
            let lse = mx + log(s);
            for (var i = 0u; i < p.n; i = i + 1u) { l[i * p.n + j] = l[i * p.n + j] - lse; }
        }
        for (var i = 0u; i < nn; i = i + 1u) { traj[((idx * 2u * p.iters) + 2u * it + 1u) * nn + i] = l[i]; }
    }
    var gr: array<f32, 36>;
    for (var i = 0u; i < nn; i = i + 1u) { gr[i] = gy[base + i] * y[base + i]; }
    for (var q = 0u; q < p.iters; q = q + 1u) {
        let it = p.iters - 1u - q;
        let cb = ((idx * 2u * p.iters) + 2u * it + 1u) * nn;
        for (var j = 0u; j < p.n; j = j + 1u) {
            var s = 0.0;
            for (var i = 0u; i < p.n; i = i + 1u) { s = s + gr[i * p.n + j]; }
            for (var i = 0u; i < p.n; i = i + 1u) { gr[i * p.n + j] = gr[i * p.n + j] - exp(traj[cb + i * p.n + j]) * s; }
        }
        let rb = ((idx * 2u * p.iters) + 2u * it) * nn;
        for (var i = 0u; i < p.n; i = i + 1u) {
            var s = 0.0;
            for (var j = 0u; j < p.n; j = j + 1u) { s = s + gr[i * p.n + j]; }
            for (var j = 0u; j < p.n; j = j + 1u) { gr[i * p.n + j] = gr[i * p.n + j] - exp(traj[rb + i * p.n + j]) * s; }
        }
    }
    for (var i = 0u; i < nn; i = i + 1u) { dl[base + i] = dl[base + i] + gr[i]; }
}

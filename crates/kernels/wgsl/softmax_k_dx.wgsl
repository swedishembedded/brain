// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Softmax-over-strided-K backward.
//   y  : [N, K, M]   the FORWARD OUTPUT (probabilities)
//   dy : [N, K, M]
//   dx : [N, K, M]   read_write
//
//   dx_k = y_k * (dy_k - sum_j dy_j * y_j)
//
// Takes the forward OUTPUT, matching softmax_hw_dx and brain's attention
// softmax backward: the Jacobian is expressed purely in the probabilities, so
// recomputing exp/max here would be slower AND a second place for the two to
// disagree.
//
// One invocation per (n, m) group — the sum_j term couples all K entries, so a
// per-element invocation would redo the same reduction K times.

struct Params {
    N: u32,
    K: u32,
    M: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       y:  array<f32>;
@group(0) @binding(2) var<storage, read>       dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.N * p.M;
    if (idx >= total) { return; }
    let m = idx % p.M;
    let n = idx / p.M;
    let base = n * p.K * p.M + m;

    var dot = 0.0;
    for (var k: u32 = 0u; k < p.K; k = k + 1u) {
        dot = dot + dy[base + k * p.M] * y[base + k * p.M];
    }
    for (var k: u32 = 0u; k < p.K; k = k + 1u) {
        dx[base + k * p.M] = y[base + k * p.M] * (dy[base + k * p.M] - dot);
    }
}

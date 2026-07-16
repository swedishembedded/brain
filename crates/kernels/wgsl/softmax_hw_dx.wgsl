// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Softmax-over-H*W backward.
//   y  : [N, 1, H, W]  the FORWARD OUTPUT (softmax probabilities)
//   dy : [N, 1, H, W]
//   dx : [N, 1, H, W]   read_write
//
//   dx_i = y_i * (dy_i - sum_j dy_j * y_j)
//
// Takes the forward OUTPUT rather than its input — the standard softmax-backward
// convention (cf. brain's attn_softmax backward), since the Jacobian is expressed
// purely in terms of the probabilities and recomputing exp/max here would be both
// slower and a second place for the two to disagree.
//
// One invocation per IMAGE, serial over H*W: the sum_j term couples every element
// to every other, so a per-element invocation would recompute the same reduction
// H*W times.

struct Params {
    N: u32,
    HW: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       y:  array<f32>;
@group(0) @binding(2) var<storage, read>       dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let n = gidx;
    if (n >= p.N) { return; }
    let base = n * p.HW;

    var dot = 0.0;
    for (var i: u32 = 0u; i < p.HW; i = i + 1u) {
        dot = dot + dy[base + i] * y[base + i];
    }
    for (var i: u32 = 0u; i < p.HW; i = i + 1u) {
        dx[base + i] = y[base + i] * (dy[base + i] - dot);
    }
}

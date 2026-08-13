// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Load-balancing fractions used by the aux-loss gradient
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Load-balancing fractions used by the aux-loss gradient:
//   f_e = (number of tokens routed to expert e) / (n_rows * top_k)
// One invocation per expert (selection = gate[*,e] > 0).

struct Params {
    n_rows: u32,
    n_experts: u32,
    top_k: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       gate: array<f32>;
@group(0) @binding(2) var<storage, read_write> fe:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let e = gidx;
    if (e >= p.n_experts) { return; }
    var count = 0.0;
    for (var n: u32 = 0u; n < p.n_rows; n = n + 1u) {
        if (gate[n * p.n_experts + e] > 0.0) { count = count + 1.0; }
    }
    fe[e] = count / f32(p.n_rows * p.top_k);
}

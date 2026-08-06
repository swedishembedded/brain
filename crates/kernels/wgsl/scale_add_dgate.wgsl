// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  MoE combine backward, part 2 — gradient w.r.t
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// MoE combine backward, part 2 — gradient w.r.t. the gate weight of one expert:
//   d_gate[t, e_idx] = sum_c expert_out_e[t,c] * d_moe_acc[t,c]
// One invocation per token row; writes a single column of d_gate.

struct Params {
    n_rows: u32,
    d_model: u32,
    n_experts: u32,
    e_idx: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       expert_out: array<f32>;
@group(0) @binding(2) var<storage, read>       d_moe_acc:  array<f32>;
@group(0) @binding(3) var<storage, read_write> d_gate:     array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let t = gidx;
    if (t >= p.n_rows) { return; }
    let base = t * p.d_model;
    var acc = 0.0;
    for (var c: u32 = 0u; c < p.d_model; c = c + 1u) {
        acc = acc + expert_out[base + c] * d_moe_acc[base + c];
    }
    d_gate[t * p.n_experts + p.e_idx] = acc;
}

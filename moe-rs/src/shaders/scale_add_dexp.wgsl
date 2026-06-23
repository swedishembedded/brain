// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// MoE combine backward, part 1 — gradient w.r.t. one expert's output:
//   moe_acc[t,c] = sum_e gate[t,e] * expert_out_e[t,c]
//   => d_expert_out_e[t,c] = gate[t,e_idx] * d_moe_acc[t,c]
// Elementwise over n_rows * d_model.

struct Params {
    n_rows: u32,
    d_model: u32,
    n_experts: u32,
    e_idx: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       gate:      array<f32>;
@group(0) @binding(2) var<storage, read>       d_moe_acc: array<f32>;
@group(0) @binding(3) var<storage, read_write> d_expert:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = p.n_rows * p.d_model;
    if (idx >= total) { return; }
    let t = idx / p.d_model;
    d_expert[idx] = gate[t * p.n_experts + p.e_idx] * d_moe_acc[idx];
}

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// MoE combine for one expert:
//   accumulate == 0 :  acc[t, c]  = gate[t, e_idx] * src[t, c]   (initialise)
//   accumulate != 0 :  acc[t, c] += gate[t, e_idx] * src[t, c]
// The first expert uses "set" so the accumulator needs no separate clear;
// every token passes through expert 0 (we evaluate all experts densely), so
// this correctly zero-initialises non-selected tokens. Elementwise over
// seq_len * d_model.

struct Params {
    seq_len: u32,
    d_model: u32,
    n_experts: u32,
    e_idx: u32,
    accumulate: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       gate: array<f32>;
@group(0) @binding(2) var<storage, read>       src:  array<f32>;
@group(0) @binding(3) var<storage, read_write> acc:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.seq_len * p.d_model;
    if (idx >= total) { return; }
    let t = idx / p.d_model;
    let contrib = gate[t * p.n_experts + p.e_idx] * src[idx];
    if (p.accumulate == 0u) {
        acc[idx] = contrib;
    } else {
        acc[idx] = acc[idx] + contrib;
    }
}

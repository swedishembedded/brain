// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Embedding gather: x[t, c] = emb[token[t], c]
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// Embedding gather: x[t, c] = emb[token[t], c].
// One invocation per output element (seq_len * d_model).

struct Params {
    d_model: u32,
    seq_len: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       tokens: array<u32>;
@group(0) @binding(2) var<storage, read>       emb:    array<f32>;
@group(0) @binding(3) var<storage, read_write> x:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.seq_len * p.d_model;
    if (idx >= total) { return; }
    let t = idx / p.d_model;
    let c = idx % p.d_model;
    x[idx] = emb[tokens[t] * p.d_model + c];
}

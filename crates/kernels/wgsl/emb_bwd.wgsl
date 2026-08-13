// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Embedding backward (also the tied lm_head's weight)
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Embedding backward (also the tied lm_head's weight): scatter the residual-
// stream gradient back into the rows that were looked up.
//   grad_emb[v, c] += sum_{n : token[n] == v} d_x[n, c]
// One invocation per (vocab row v, channel c); it loops the tokens so there is
// no race and no need for atomics. Accumulates onto grad_emb, which already
// holds the lm_head contribution.

struct Params {
    n_rows: u32,
    d_model: u32,
    vocab: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       tokens:   array<u32>;
@group(0) @binding(2) var<storage, read>       d_x:      array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_emb: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.vocab * p.d_model;
    if (idx >= total) { return; }
    let v = idx / p.d_model;
    let c = idx % p.d_model;
    var acc = 0.0;
    for (var n: u32 = 0u; n < p.n_rows; n = n + 1u) {
        if (tokens[n] == v) {
            acc = acc + d_x[n * p.d_model + c];
        }
    }
    grad_emb[idx] = grad_emb[idx] + acc;
}

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Row-wise cross-attention softmax over the encoder key axis
// @how   one thread per output element, 3 nested serial reductions
// @opt   1
// @cpu   native
// @gpu   yes
// @npu   yes
// @quant none
//
// Row-wise cross-attention softmax over the encoder key axis. One invocation per
// (b,h,i): normalises scores[b,h,i, 0..T_enc] into probs (non-causal). The row
// length is T_enc (keys), which differs from the query count T_dec.
// scores/probs layout: ((b*H + h)*T_dec + i)*T_enc + j.

struct Params {
    bsz: u32,
    n_heads: u32,
    t_dec: u32,   // query length (row count)
    t_enc: u32,   // key length (row width)
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       scores: array<f32>;
@group(0) @binding(2) var<storage, read_write> probs:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let Tq = p.t_dec;
    let Tk = p.t_enc;
    let total = p.bsz * p.n_heads * Tq;
    let idx = gidx;
    if (idx >= total) { return; }

    let i = idx % Tq;
    let r = idx / Tq;          // b*H + h
    let base = (r * Tq + i) * Tk;

    var mx = -3.4e38;
    for (var j: u32 = 0u; j < Tk; j = j + 1u) { mx = max(mx, scores[base + j]); }
    var sum = 0.0;
    for (var j: u32 = 0u; j < Tk; j = j + 1u) { sum = sum + exp(scores[base + j] - mx); }
    let inv = 1.0 / sum;
    for (var j: u32 = 0u; j < Tk; j = j + 1u) {
        probs[base + j] = exp(scores[base + j] - mx) * inv;
    }
}

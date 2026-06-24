// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Cross-attention backward, step 1 — gradient through (probs @ v) and the softmax.
// One invocation per (b,h,i) over the query axis (i in [0,T_dec)):
//   d_prob_j = sum_d d_out[b,i,h,d] * v[b,j,h,d]        (j in [0,T_enc))
//   dot      = sum_j probs[b,h,i,j] * d_prob_j
//   d_score_j = probs[b,h,i,j] * (d_prob_j - dot)        (softmax jacobian)
// V comes from the ENCODER-MEMORY buffer `kv` (stride kv_stride=2*d_model,
// v region at v_off=d_model). d_out is the contiguous [B*T_dec, d_model] grad.
// scores/probs/d_scores layout: ((b*H + h)*T_dec + i)*T_enc + j.

struct Params {
    bsz: u32,
    n_heads: u32,
    t_dec: u32,
    t_enc: u32,
    head_dim: u32,
    kv_stride: u32,    // 2*d_model
    v_off: u32,        // d_model
    d_model: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       d_out:    array<f32>;
@group(0) @binding(2) var<storage, read>       kv:       array<f32>;  // encoder memory (V)
@group(0) @binding(3) var<storage, read>       probs:    array<f32>;
@group(0) @binding(4) var<storage, read_write> d_scores: array<f32>;

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
    let h = r % p.n_heads;
    let b = r / p.n_heads;
    let hd = p.head_dim;

    let p_base = (r * Tq + i) * Tk;
    let out_base = (b * Tq + i) * p.d_model + h * hd;

    var dot = 0.0;
    for (var j: u32 = 0u; j < Tk; j = j + 1u) {
        let v_base = (b * Tk + j) * p.kv_stride + p.v_off + h * hd;
        var dprob = 0.0;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            dprob = dprob + d_out[out_base + d] * kv[v_base + d];
        }
        dot = dot + probs[p_base + j] * dprob;
    }
    for (var j: u32 = 0u; j < Tk; j = j + 1u) {
        let v_base = (b * Tk + j) * p.kv_stride + p.v_off + h * hd;
        var dprob = 0.0;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            dprob = dprob + d_out[out_base + d] * kv[v_base + d];
        }
        d_scores[p_base + j] = probs[p_base + j] * (dprob - dot);
    }
}

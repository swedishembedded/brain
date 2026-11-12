// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Banded attention backward, step 1 - gradient through (probs @ v) and the softmax over the band slab
// @how   one thread per query row, two serial passes over the band, each reducing head_dim
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Banded attention backward, step 1, the twin of `attn_bwd_dscores_cross` for
// the [rows, 2*window+1] slab. One invocation per (b,h,i):
//   d_prob_s  = sum_d d_out[b,i,h,d] * v[b,j,h,d],   j = q0 + i + s - window
//   dot       = sum_s probs[b,h,i,s] * d_prob_s
//   d_score_s = probs[b,h,i,s] * (d_prob_s - dot)       (softmax jacobian)
// Slots whose key is outside [0, t_enc) get d_score 0 (their probability is 0).

struct Params {
    bsz: u32,
    n_heads: u32,
    t_dec: u32,
    t_enc: u32,
    head_dim: u32,
    kv_stride: u32,
    v_off: u32,
    d_model: u32,
    q0: u32,
    window: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       d_out:    array<f32>;
@group(0) @binding(2) var<storage, read>       kv:       array<f32>;
@group(0) @binding(3) var<storage, read>       probs:    array<f32>;
@group(0) @binding(4) var<storage, read_write> d_scores: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let W = 2u * p.window + 1u;
    let Tq = p.t_dec;
    let Tk = p.t_enc;
    let total = p.bsz * p.n_heads * Tq;
    if (idx >= total) { return; }

    let i = idx % Tq;
    let r = idx / Tq;          // b*H + h
    let h = r % p.n_heads;
    let b = r / p.n_heads;
    let hd = p.head_dim;

    let p_base = (r * Tq + i) * W;
    let out_base = (b * Tq + i) * p.d_model + h * hd;
    let j_left = i32(p.q0 + i) - i32(p.window);

    var dot = 0.0;
    for (var s: u32 = 0u; s < W; s = s + 1u) {
        let j = j_left + i32(s);
        if (j < 0 || j >= i32(Tk)) { continue; }
        let v_base = (b * Tk + u32(j)) * p.kv_stride + p.v_off + h * hd;
        var dprob = 0.0;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            dprob = dprob + d_out[out_base + d] * kv[v_base + d];
        }
        dot = dot + probs[p_base + s] * dprob;
    }
    for (var s: u32 = 0u; s < W; s = s + 1u) {
        let j = j_left + i32(s);
        if (j < 0 || j >= i32(Tk)) { d_scores[p_base + s] = 0.0; continue; }
        let v_base = (b * Tk + u32(j)) * p.kv_stride + p.v_off + h * hd;
        var dprob = 0.0;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            dprob = dprob + d_out[out_base + d] * kv[v_base + d];
        }
        d_scores[p_base + s] = probs[p_base + s] * (dprob - dot);
    }
}

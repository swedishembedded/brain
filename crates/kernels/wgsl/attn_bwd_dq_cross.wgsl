// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Cross-attention backward, step 3 - gradient w.r.t
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Cross-attention backward, step 3 — gradient w.r.t. q (decoder):
//   d_q[b,i,h,d] = scale * sum_{j<T_enc} d_score[b,h,i,j] * k[b,j,h,d]
// K comes from the ENCODER-MEMORY buffer `kv` (stride kv_stride=2*d_model,
// k_off=0). The grad is written into the q region of the DECODER grad buffer
// `d_q` (stride q_stride=3*d_model, q_off=0). d_scores layout:
// ((b*H + h)*T_dec + i)*T_enc + j. One invocation per (b,h,i,d), i over [0,T_dec).

struct Params {
    bsz: u32,
    n_heads: u32,
    t_dec: u32,
    t_enc: u32,
    head_dim: u32,
    q_stride: u32,     // 3*d_model (decoder fused QKV)
    kv_stride: u32,    // 2*d_model (encoder fused KV)
    q_off: u32,        // 0
    k_off: u32,        // 0
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       d_scores: array<f32>;
@group(0) @binding(2) var<storage, read>       kv:       array<f32>;  // encoder memory (K)
@group(0) @binding(3) var<storage, read_write> d_q:      array<f32>;  // decoder grad

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let Tq = p.t_dec;
    let Tk = p.t_enc;
    let hd = p.head_dim;
    let total = p.bsz * p.n_heads * Tq * hd;
    let idx = gidx;
    if (idx >= total) { return; }

    let d = idx % hd;
    let r1 = idx / hd;
    let i = r1 % Tq;
    let r2 = r1 / Tq;
    let h = r2 % p.n_heads;
    let b = r2 / p.n_heads;
    let scale = inverseSqrt(f32(hd));

    let s_base = ((b * p.n_heads + h) * Tq + i) * Tk;
    var acc = 0.0;
    for (var j: u32 = 0u; j < Tk; j = j + 1u) {
        let k = kv[(b * Tk + j) * p.kv_stride + p.k_off + h * hd + d];
        acc = acc + d_scores[s_base + j] * k;
    }
    d_q[(b * Tq + i) * p.q_stride + p.q_off + h * hd + d] = acc * scale;
}

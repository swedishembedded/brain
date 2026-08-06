// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Accumulating twin of attn_bwd_dv_cross for QUERY-CHUNKED backward
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Accumulating twin of attn_bwd_dv_cross for QUERY-CHUNKED backward: d_v sums
// over ALL query rows; `acc_flag = 0` assigns (first chunk), `1` accumulates.
// Cross-attention backward, step 2 — gradient w.r.t. v (encoder memory):
//   d_v[b,j,h,d] = sum_{i<T_dec} probs[b,h,i,j] * d_out[b,i,h,d]
// Non-causal: sum over all query positions i. Written into the v region of the
// ENCODER-MEMORY grad buffer `d_kv` (stride kv_stride=2*d_model, v_off=d_model).
// probs layout: ((b*H + h)*T_dec + i)*T_enc + j. One invocation per (b,h,j,d),
// j over the key axis [0,T_enc).

struct Params {
    bsz: u32,
    n_heads: u32,
    t_dec: u32,
    t_enc: u32,
    head_dim: u32,
    kv_stride: u32,    // 2*d_model
    v_off: u32,        // d_model
    d_model: u32,
    acc_flag: u32,     // 0 = assign (first chunk), 1 = accumulate
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       probs: array<f32>;
@group(0) @binding(2) var<storage, read>       d_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> d_kv:  array<f32>;  // encoder-memory grad

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let Tq = p.t_dec;
    let Tk = p.t_enc;
    let hd = p.head_dim;
    let total = p.bsz * p.n_heads * Tk * hd;
    let idx = gidx;
    if (idx >= total) { return; }

    let d = idx % hd;
    let r1 = idx / hd;
    let j = r1 % Tk;
    let r2 = r1 / Tk;
    let h = r2 % p.n_heads;
    let b = r2 / p.n_heads;

    var acc = 0.0;
    for (var i: u32 = 0u; i < Tq; i = i + 1u) {
        let prob = probs[((b * p.n_heads + h) * Tq + i) * Tk + j];
        acc = acc + prob * d_out[(b * Tq + i) * p.d_model + h * hd + d];
    }
    let o = (b * Tk + j) * p.kv_stride + p.v_off + h * hd + d;
    let prev = select(0.0, d_kv[o], p.acc_flag == 1u);
    d_kv[o] = prev + acc;
}

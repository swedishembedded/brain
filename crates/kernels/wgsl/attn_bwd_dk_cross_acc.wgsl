// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Accumulating twin of attn_bwd_dk_cross for QUERY-CHUNKED backward
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Accumulating twin of attn_bwd_dk_cross for QUERY-CHUNKED backward: d_k sums
// over ALL query rows, so each chunk contributes a partial sum — `acc_flag = 0`
// on the first chunk assigns, `1` on later chunks adds. (The assigning original
// forces chunk == full span; this kernel is what makes 8k-training chunkable.)
// Original contract:
//   d_k[b,j,h,d] = scale * sum_{i<T_dec} d_score[b,h,i,j] * q[b,i,h,d]
// Q comes from the DECODER buffer `q` (stride q_stride=3*d_model, q_off=0). The
// grad is written into the k region of the ENCODER-MEMORY grad buffer `d_kv`
// (stride kv_stride=2*d_model, k_off=0). d_scores layout:
// ((b*H + h)*T_dec + i)*T_enc + j. One invocation per (b,h,j,d), j over [0,T_enc).

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
    acc_flag: u32,     // 0 = assign (first chunk), 1 = accumulate
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       d_scores: array<f32>;
@group(0) @binding(2) var<storage, read>       q:        array<f32>;  // decoder buffer (Q)
@group(0) @binding(3) var<storage, read_write> d_kv:     array<f32>;  // encoder-memory grad

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
    let scale = inverseSqrt(f32(hd));

    var acc = 0.0;
    for (var i: u32 = 0u; i < Tq; i = i + 1u) {
        let s = d_scores[((b * p.n_heads + h) * Tq + i) * Tk + j];
        let qv = q[(b * Tq + i) * p.q_stride + p.q_off + h * hd + d];
        acc = acc + s * qv;
    }
    let o = (b * Tk + j) * p.kv_stride + p.k_off + h * hd + d;
    let prev = select(0.0, d_kv[o], p.acc_flag == 1u);
    d_kv[o] = prev + acc * scale;
}

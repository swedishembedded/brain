// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Cross-attention backward, step 1, one WORKGROUP per query row - cooperative twin of attn_bwd_dscores_cross
// @how   64-thread workgroup tile, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Cross-attention twin of attn_bwd_dscores_rows.wgsl - the key/value axis is
// the ENCODER length T_enc, separate from the query axis T_dec, and V comes
// from the encoder-memory buffer `kv` (stride kv_stride, v region at v_off).
// `model::vit::cross_q_bwd` also dispatches this family for plain (bidir)
// self-attention with q == kv, so this one physical kernel covers both
// regimes for every model that shares that builder. See
// attn_bwd_dscores_rows.wgsl for the coalescing analysis and the
// reduction-order note. Dispatch: (bsz * n_heads * t_dec) * 64 invocations.

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

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear workgroup index (identity for 1D dispatch).
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    let Tq = p.t_dec;
    let Tk = p.t_enc;
    let total = p.bsz * p.n_heads * Tq;
    if (row >= total) { return; }

    let i = row % Tq;
    let r = row / Tq;          // b*n_heads + h
    let h = r % p.n_heads;
    let b = r / p.n_heads;
    let hd = p.head_dim;

    let p_base = (r * Tq + i) * Tk;
    let out_base = (b * Tq + i) * p.d_model + h * hd;

    var acc = 0.0;
    for (var j: u32 = t; j < Tk; j = j + 64u) {
        let v_base = (b * Tk + j) * p.kv_stride + p.v_off + h * hd;
        var dprob = 0.0;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            dprob = dprob + d_out[out_base + d] * kv[v_base + d];
        }
        acc = acc + probs[p_base + j] * dprob;
    }
    partial[t] = acc;
    workgroupBarrier();
    var dot = 0.0;
    for (var k: u32 = 0u; k < 64u; k = k + 1u) {
        dot = dot + partial[k];
    }

    for (var j: u32 = t; j < Tk; j = j + 64u) {
        let v_base = (b * Tk + j) * p.kv_stride + p.v_off + h * hd;
        var dprob = 0.0;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            dprob = dprob + d_out[out_base + d] * kv[v_base + d];
        }
        d_scores[p_base + j] = probs[p_base + j] * (dprob - dot);
    }
}

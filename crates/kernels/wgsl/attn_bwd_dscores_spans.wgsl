// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Softmax-jacobian score gradient over RAGGED spans in one dispatch
// @how   64-thread workgroup tile, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// `attn_bwd_dscores_cross_rows` for the ragged-span layout, and the same
// cooperative shape: ONE WORKGROUP per score row, because every element of a
// row needs the same `sum_k p_ik (d_ctx_i . v_k)` and computing it per element
// is what made the per-element twin cost more than every GEMM in the reverse
// pass put together. See `attn_scores_spans` for why this family exists, for
// the work table, and for the shared `Params`.
//
// The table entry here is counted in score ROWS, one per workgroup, so
// `elem0` IS this workgroup's row index within the span's `heads * len` rows.
//
//   d_scores[s,h,i,j] = p[i,j] * ((d_ctx_i . v_j) - sum_k p[i,k] (d_ctx_i . v_k))
//
// The reduction is the same two-pass, 64-partial, in-order sum the per-row
// cross kernel uses, so the two agree to the last bit of rounding order.

struct Params {
    n_wg: u32,
    n_heads: u32,
    head_dim: u32,
    qkv_stride: u32,
    q_off: u32,
    k_off: u32,
    v_off: u32,
    d_model: u32,      // d_ctx row stride
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       work:     array<u32>;
@group(0) @binding(2) var<storage, read>       d_ctx:    array<f32>;
@group(0) @binding(3) var<storage, read>       qkv:      array<f32>;  // V region
@group(0) @binding(4) var<storage, read>       probs:    array<f32>;
@group(0) @binding(5) var<storage, read_write> d_scores: array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear workgroup index (identity for 1D dispatch). This
    // kernel HAS a barrier, so it is a work-group kernel to both backends and
    // may read the workgroup builtins - which the per-element members of this
    // family deliberately do not.
    let w = wgid.y * nwg.x + wgid.x;
    if (w >= p.n_wg) { return; }
    let e4 = 4u * w;
    let row0 = work[e4];
    let len = work[e4 + 1u];
    let sbase = work[e4 + 2u];
    let row = work[e4 + 3u];
    let t = lid.x;

    let h = row / len;
    let i = row % len;
    let hd = p.head_dim;
    let p_base = sbase + row * len;
    let out_base = (row0 + i) * p.d_model + h * hd;

    var acc = 0.0;
    for (var j: u32 = t; j < len; j = j + 64u) {
        let v_base = (row0 + j) * p.qkv_stride + p.v_off + h * hd;
        var dprob = 0.0;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            dprob = dprob + d_ctx[out_base + d] * qkv[v_base + d];
        }
        acc = acc + probs[p_base + j] * dprob;
    }
    partial[t] = acc;
    workgroupBarrier();
    var dot = 0.0;
    for (var k: u32 = 0u; k < 64u; k = k + 1u) {
        dot = dot + partial[k];
    }

    for (var j: u32 = t; j < len; j = j + 64u) {
        let v_base = (row0 + j) * p.qkv_stride + p.v_off + h * hd;
        var dprob = 0.0;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            dprob = dprob + d_ctx[out_base + d] * qkv[v_base + d];
        }
        d_scores[p_base + j] = probs[p_base + j] * (dprob - dot);
    }
}

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  GQA attention backward, step 1 - gradient through (probs @ v) and softmax
// @how   one thread per output element, 3 nested serial reductions
// @opt   1
// @cpu   native
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// GQA attention backward, step 1 — gradient through (probs @ v) and softmax.
// One invocation per (b,h,i):
//   d_prob_j = sum_d d_ctx[b,i,h,d] * v[b,j,hkv,d]        (j <= i, hkv = h/group)
//   dot      = sum_j probs[b,h,i,j] * d_prob_j
//   d_score_j = probs[b,h,i,j] * (d_prob_j - dot)          (softmax jacobian)
// d_ctx is the contiguous [B*T, n_heads*head_dim] grad; v is the separate
// [B*T, n_kv_heads*head_dim] buffer.

struct Params {
    bsz: u32,
    n_heads: u32,
    n_kv_heads: u32,
    tcols: u32,       // T
    head_dim: u32,
    group: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       d_ctx:    array<f32>;
@group(0) @binding(2) var<storage, read>       v:        array<f32>;
@group(0) @binding(3) var<storage, read>       probs:    array<f32>;
@group(0) @binding(4) var<storage, read_write> d_scores: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let T = p.tcols;
    let total = p.bsz * p.n_heads * T;
    let idx = gidx;
    if (idx >= total) { return; }

    let i = idx % T;
    let r = idx / T;          // b*n_heads + h
    let h = r % p.n_heads;
    let b = r / p.n_heads;
    let hd = p.head_dim;
    let hkv = h / p.group;
    let k_row = p.n_kv_heads * hd;
    let q_row = p.n_heads * hd;

    let p_base = (r * T + i) * T;
    let ctx_base = (b * T + i) * q_row + h * hd;

    var dot = 0.0;
    for (var j: u32 = 0u; j <= i; j = j + 1u) {
        let v_base = (b * T + j) * k_row + hkv * hd;
        var dprob = 0.0;
        for (var d: u32 = 0u; d < hd; d = d + 1u) {
            dprob = dprob + d_ctx[ctx_base + d] * v[v_base + d];
        }
        dot = dot + probs[p_base + j] * dprob;
    }
    for (var j: u32 = 0u; j < T; j = j + 1u) {
        if (j <= i) {
            let v_base = (b * T + j) * k_row + hkv * hd;
            var dprob = 0.0;
            for (var d: u32 = 0u; d < hd; d = d + 1u) {
                dprob = dprob + d_ctx[ctx_base + d] * v[v_base + d];
            }
            d_scores[p_base + j] = probs[p_base + j] * (dprob - dot);
        } else {
            d_scores[p_base + j] = 0.0;
        }
    }
}

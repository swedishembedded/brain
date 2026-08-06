// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  GQA attention backward, step 3 — gradient w.r.t
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// GQA attention backward, step 3 — gradient w.r.t. q (post-RoPE):
//   d_q[b,i,h,d] = scale * sum_{j<=i} d_score[b,h,i,j] * k[b,j,hkv,d]
// scale = 1/sqrt(head_dim), hkv = h/group. Written into the q-grad buffer
// [B*T, n_heads*head_dim]; k is the separate [B*T, n_kv_heads*head_dim] buffer.
// One invocation per (b,h,i,d).

struct Params {
    bsz: u32,
    n_heads: u32,
    n_kv_heads: u32,
    tcols: u32,
    head_dim: u32,
    group: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       d_scores: array<f32>;
@group(0) @binding(2) var<storage, read>       k:        array<f32>;
@group(0) @binding(3) var<storage, read_write> d_q:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let T = p.tcols;
    let hd = p.head_dim;
    let total = p.bsz * p.n_heads * T * hd;
    let idx = gidx;
    if (idx >= total) { return; }

    let d = idx % hd;
    let r1 = idx / hd;
    let i = r1 % T;
    let r2 = r1 / T;
    let h = r2 % p.n_heads;
    let b = r2 / p.n_heads;
    let scale = inverseSqrt(f32(hd));
    let hkv = h / p.group;
    let k_row = p.n_kv_heads * hd;
    let q_row = p.n_heads * hd;

    let s_base = ((b * p.n_heads + h) * T + i) * T;
    var acc = 0.0;
    for (var j: u32 = 0u; j <= i; j = j + 1u) {
        let kk = k[(b * T + j) * k_row + hkv * hd + d];
        acc = acc + d_scores[s_base + j] * kk;
    }
    d_q[(b * T + i) * q_row + h * hd + d] = acc * scale;
}

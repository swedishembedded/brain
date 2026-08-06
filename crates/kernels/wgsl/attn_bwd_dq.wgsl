// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Attention backward, step 3 — gradient w.r.t
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Attention backward, step 3 — gradient w.r.t. q (post-RoPE):
//   d_q[b,i,h,d] = scale * sum_{j<=i} d_score[b,h,i,j] * k[b,j,h,d]
// Written into the q region of d_qkv. One invocation per (b,h,i,d).

struct Params {
    bsz: u32,
    n_heads: u32,
    tcols: u32,
    head_dim: u32,
    qkv_stride: u32,
    q_off: u32,
    k_off: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       d_scores: array<f32>;
@group(0) @binding(2) var<storage, read>       qkv:      array<f32>;
@group(0) @binding(3) var<storage, read_write> d_qkv:    array<f32>;

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

    let s_base = ((b * p.n_heads + h) * T + i) * T;
    var acc = 0.0;
    for (var j: u32 = 0u; j <= i; j = j + 1u) {
        let k = qkv[(b * T + j) * p.qkv_stride + p.k_off + h * hd + d];
        acc = acc + d_scores[s_base + j] * k;
    }
    d_qkv[(b * T + i) * p.qkv_stride + p.q_off + h * hd + d] = acc * scale;
}

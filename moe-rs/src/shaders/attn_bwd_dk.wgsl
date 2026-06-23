// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Attention backward, step 4 — gradient w.r.t. k (post-RoPE):
//   d_k[b,j,h,d] = scale * sum_{i>=j} d_score[b,h,i,j] * q[b,i,h,d]
// Written into the k region of d_qkv. One invocation per (b,h,j,d).

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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let T = p.tcols;
    let hd = p.head_dim;
    let total = p.bsz * p.n_heads * T * hd;
    let idx = gid.x;
    if (idx >= total) { return; }

    let d = idx % hd;
    let r1 = idx / hd;
    let j = r1 % T;
    let r2 = r1 / T;
    let h = r2 % p.n_heads;
    let b = r2 / p.n_heads;
    let scale = inverseSqrt(f32(hd));

    var acc = 0.0;
    for (var i: u32 = j; i < T; i = i + 1u) {
        let s = d_scores[((b * p.n_heads + h) * T + i) * T + j];
        let q = qkv[(b * T + i) * p.qkv_stride + p.q_off + h * hd + d];
        acc = acc + s * q;
    }
    d_qkv[(b * T + j) * p.qkv_stride + p.k_off + h * hd + d] = acc * scale;
}

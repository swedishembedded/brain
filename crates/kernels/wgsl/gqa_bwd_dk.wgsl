// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  GQA attention backward, step 4 — gradient w.r.t
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// GQA attention backward, step 4 — gradient w.r.t. k (post-RoPE), accumulated
// over the query-head group that shares each kv head:
//   d_k[b,j,hkv,d] = scale * sum_{h in group(hkv)} sum_{i>=j}
//                                   d_score[b,h,i,j] * q[b,i,h,d]
// scale = 1/sqrt(head_dim). Written into the k-grad buffer
// [B*T, n_kv_heads*head_dim]; q is the [B*T, n_heads*head_dim] buffer.
// One invocation per (b,hkv,j,d) — owns one output element, so no atomics.

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
@group(0) @binding(2) var<storage, read>       q:        array<f32>;
@group(0) @binding(3) var<storage, read_write> d_k:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let T = p.tcols;
    let hd = p.head_dim;
    let total = p.bsz * p.n_kv_heads * T * hd;
    let idx = gidx;
    if (idx >= total) { return; }

    let d = idx % hd;
    let r1 = idx / hd;
    let j = r1 % T;
    let r2 = r1 / T;
    let hkv = r2 % p.n_kv_heads;
    let b = r2 / p.n_kv_heads;
    let scale = inverseSqrt(f32(hd));
    let q_row = p.n_heads * hd;
    let k_row = p.n_kv_heads * hd;

    var acc = 0.0;
    for (var gi: u32 = 0u; gi < p.group; gi = gi + 1u) {
        let h = hkv * p.group + gi;
        let sh = (b * p.n_heads + h) * T;
        for (var i: u32 = j; i < T; i = i + 1u) {
            acc = acc + d_scores[(sh + i) * T + j] * q[(b * T + i) * q_row + h * hd + d];
        }
    }
    d_k[(b * T + j) * k_row + hkv * hd + d] = acc * scale;
}

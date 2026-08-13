// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Backward w.r.t. k for the biased/configurable-scale scores kernels (attn_scores_{bidir,causal}_bias)
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Backward w.r.t. k for the biased/configurable-scale scores kernels
// (attn_scores_{bidir,causal}_bias):
//   d_k[b,j,h,d] = scale * sum_i d_score[b,h,i,j] * q[b,i,h,d]
// `causal != 0` restricts the sum to i >= j (temporal attention); otherwise all
// i (spatial). `scale` is the same caller constant used in the forward.
// Written into the k region of d_qkv. One invocation per (b,h,j,d).

struct Params {
    bsz: u32,
    n_heads: u32,
    tcols: u32,
    head_dim: u32,
    qkv_stride: u32,
    q_off: u32,
    k_off: u32,
    scale: u32,        // f32 bits
    causal: u32,       // 0 = bidir, else i>=j
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       d_scores: array<f32>;
@group(0) @binding(2) var<storage, read>       qkv:      array<f32>;
@group(0) @binding(3) var<storage, read_write> d_qkv:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let T = p.tcols;
    let hd = p.head_dim;
    let total = p.bsz * p.n_heads * T * hd;
    let idx = gidx;
    if (idx >= total) { return; }

    let d = idx % hd;
    let r1 = idx / hd;
    let j = r1 % T;
    let r2 = r1 / T;
    let h = r2 % p.n_heads;
    let b = r2 / p.n_heads;

    var istart = 0u;
    if (p.causal != 0u) { istart = j; }

    var acc = 0.0;
    for (var i: u32 = istart; i < T; i = i + 1u) {
        let s = d_scores[((b * p.n_heads + h) * T + i) * T + j];
        let q = qkv[(b * T + i) * p.qkv_stride + p.q_off + h * hd + d];
        acc = acc + s * q;
    }
    d_qkv[(b * T + j) * p.qkv_stride + p.k_off + h * hd + d] = acc * bitcast<f32>(p.scale);
}

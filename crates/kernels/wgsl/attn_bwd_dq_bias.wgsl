// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Backward w.r.t. q for the biased/configurable-scale scores kernels (attn_scores_{bidir,causal}_bias)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Backward w.r.t. q for the biased/configurable-scale scores kernels
// (attn_scores_{bidir,causal}_bias). The additive bias does not depend on q, so
// its gradient path is only through the dot product:
//   d_q[b,i,h,d] = scale * sum_j d_score[b,h,i,j] * k[b,j,h,d]
// `causal != 0` restricts the sum to j <= i (temporal attention); otherwise all
// j (spatial). `scale` is the same caller constant used in the forward.
// Written into the q region of d_qkv. One invocation per (b,h,i,d).

struct Params {
    bsz: u32,
    n_heads: u32,
    tcols: u32,
    head_dim: u32,
    qkv_stride: u32,
    q_off: u32,
    k_off: u32,
    scale: u32,        // f32 bits
    causal: u32,       // 0 = bidir, else j<=i
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
    let i = r1 % T;
    let r2 = r1 / T;
    let h = r2 % p.n_heads;
    let b = r2 / p.n_heads;

    var jmax = T;
    if (p.causal != 0u) { jmax = i + 1u; }

    let s_base = ((b * p.n_heads + h) * T + i) * T;
    var acc = 0.0;
    for (var j: u32 = 0u; j < jmax; j = j + 1u) {
        let k = qkv[(b * T + j) * p.qkv_stride + p.k_off + h * hd + d];
        acc = acc + d_scores[s_base + j] * k;
    }
    d_qkv[(b * T + i) * p.qkv_stride + p.q_off + h * hd + d] = acc * bitcast<f32>(p.scale);
}

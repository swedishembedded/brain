// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  MLA (Multi-head Latent Attention) scores (forward), for GLM-5.2
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// MLA (Multi-head Latent Attention) scores (forward), for GLM-5.2.
//   scores[b,h,i,j] = scale * ( sum_dn q_pass[b,i,h,dn]*k_pass[b,j,h,dn]
//                             + sum_dr q_rot[b,i,h,dr]*k_rot[b,j,dr] )   for j<=i
//                   = -inf                                                for j>i
// where scale = 1/sqrt(nope+rope). `q_pass`/`k_pass` are contiguous
// [B*T, H*nope]; `q_rot` is [B*T, H*rope]; `k_rot` is the shared (MQA) rope key
// [B*T, rope] (one head, broadcast over all H). scores layout: ((b*H+h)*T+i)*T+j.
// One invocation per (b,h,i,j).

struct Params {
    bsz: u32,
    n_heads: u32,
    tcols: u32,   // T
    nope: u32,    // qk_nope_head_dim
    rope: u32,    // qk_rope_head_dim
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q_pass: array<f32>;
@group(0) @binding(2) var<storage, read>       q_rot:  array<f32>;
@group(0) @binding(3) var<storage, read>       k_pass: array<f32>;
@group(0) @binding(4) var<storage, read>       k_rot:  array<f32>;
@group(0) @binding(5) var<storage, read_write> scores: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let T = p.tcols;
    let total = p.bsz * p.n_heads * T * T;
    if (gidx >= total) { return; }

    let j = gidx % T;
    let r1 = gidx / T;
    let i = r1 % T;
    let r2 = r1 / T;
    let h = r2 % p.n_heads;
    let b = r2 / p.n_heads;

    if (j > i) { scores[gidx] = -3.4e38; return; }

    let np = p.nope;
    let rp = p.rope;
    let qp_base = (b * T + i) * p.n_heads * np + h * np;
    let kp_base = (b * T + j) * p.n_heads * np + h * np;
    let qr_base = (b * T + i) * p.n_heads * rp + h * rp;
    let kr_base = (b * T + j) * rp;

    var s = 0.0;
    for (var d: u32 = 0u; d < np; d = d + 1u) {
        s = s + q_pass[qp_base + d] * k_pass[kp_base + d];
    }
    for (var d: u32 = 0u; d < rp; d = d + 1u) {
        s = s + q_rot[qr_base + d] * k_rot[kr_base + d];
    }
    scores[gidx] = s * inverseSqrt(f32(np + rp));
}

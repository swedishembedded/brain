// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  MLA backward — grad w.r.t
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// MLA backward — grad w.r.t. the query nope block `q_pass` (post-projection):
//   d_q_pass[b,i,h,dn] = scale * sum_{j<=i} d_scores[b,h,i,j] * k_pass[b,j,h,dn]
// scale = 1/sqrt(nope+rope); `d_scores` is grad of the scaled pre-softmax scores.
// `q_pass`/`k_pass` are contiguous [B*T, H*nope]. One invocation per (b,h,i,dn).

struct Params {
    bsz: u32,
    n_heads: u32,
    tcols: u32,
    nope: u32,
    rope: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       d_scores: array<f32>;
@group(0) @binding(2) var<storage, read>       k_pass:   array<f32>;
@group(0) @binding(3) var<storage, read_write> d_q_pass: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let T = p.tcols;
    let np = p.nope;
    let total = p.bsz * p.n_heads * T * np;
    if (gidx >= total) { return; }

    let dn = gidx % np;
    let r1 = gidx / np;
    let i = r1 % T;
    let r2 = r1 / T;
    let h = r2 % p.n_heads;
    let b = r2 / p.n_heads;
    let scale = inverseSqrt(f32(np + p.rope));

    let s_base = ((b * p.n_heads + h) * T + i) * T;
    var acc = 0.0;
    for (var j: u32 = 0u; j <= i; j = j + 1u) {
        let k = k_pass[(b * T + j) * p.n_heads * np + h * np + dn];
        acc = acc + d_scores[s_base + j] * k;
    }
    d_q_pass[(b * T + i) * p.n_heads * np + h * np + dn] = acc * scale;
}

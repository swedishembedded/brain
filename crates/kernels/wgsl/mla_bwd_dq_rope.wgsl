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
// MLA backward — grad w.r.t. the query rope block `q_rot` (post-RoPE):
//   d_q_rot[b,i,h,dr] = scale * sum_{j<=i} d_scores[b,h,i,j] * k_rot[b,j,dr]
// scale = 1/sqrt(nope+rope). `q_rot` is [B*T, H*rope]; `k_rot` is the shared
// (MQA) rope key [B*T, rope]. One invocation per (b,h,i,dr).

struct Params {
    bsz: u32,
    n_heads: u32,
    tcols: u32,
    nope: u32,
    rope: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       d_scores: array<f32>;
@group(0) @binding(2) var<storage, read>       k_rot:    array<f32>;
@group(0) @binding(3) var<storage, read_write> d_q_rot:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let T = p.tcols;
    let rp = p.rope;
    let total = p.bsz * p.n_heads * T * rp;
    if (gidx >= total) { return; }

    let dr = gidx % rp;
    let r1 = gidx / rp;
    let i = r1 % T;
    let r2 = r1 / T;
    let h = r2 % p.n_heads;
    let b = r2 / p.n_heads;
    let scale = inverseSqrt(f32(p.nope + rp));

    let s_base = ((b * p.n_heads + h) * T + i) * T;
    var acc = 0.0;
    for (var j: u32 = 0u; j <= i; j = j + 1u) {
        acc = acc + d_scores[s_base + j] * k_rot[(b * T + j) * rp + dr];
    }
    d_q_rot[(b * T + i) * p.n_heads * rp + h * rp + dr] = acc * scale;
}

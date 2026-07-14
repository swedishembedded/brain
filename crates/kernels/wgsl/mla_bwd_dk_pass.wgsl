// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// MLA backward — grad w.r.t. the key nope block `k_pass`:
//   d_k_pass[b,j,h,dn] = scale * sum_{i>=j} d_scores[b,h,i,j] * q_pass[b,i,h,dn]
// scale = 1/sqrt(nope+rope). Contiguous [B*T, H*nope]. One invocation per (b,h,j,dn).

struct Params {
    bsz: u32,
    n_heads: u32,
    tcols: u32,
    nope: u32,
    rope: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       d_scores: array<f32>;
@group(0) @binding(2) var<storage, read>       q_pass:   array<f32>;
@group(0) @binding(3) var<storage, read_write> d_k_pass: array<f32>;

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
    let j = r1 % T;
    let r2 = r1 / T;
    let h = r2 % p.n_heads;
    let b = r2 / p.n_heads;
    let scale = inverseSqrt(f32(np + p.rope));

    var acc = 0.0;
    for (var i: u32 = j; i < T; i = i + 1u) {
        let s = d_scores[((b * p.n_heads + h) * T + i) * T + j];
        let q = q_pass[(b * T + i) * p.n_heads * np + h * np + dn];
        acc = acc + s * q;
    }
    d_k_pass[(b * T + j) * p.n_heads * np + h * np + dn] = acc * scale;
}

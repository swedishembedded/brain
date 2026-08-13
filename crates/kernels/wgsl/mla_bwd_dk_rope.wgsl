// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  MLA backward - grad w.r.t
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// MLA backward — grad w.r.t. the shared (MQA) rope key `k_rot`:
//   d_k_rot[b,j,dr] = scale * sum_{i>=j} sum_h d_scores[b,h,i,j] * q_rot[b,i,h,dr]
// The sum over heads is because a single `k_rot` is broadcast to every head.
// scale = 1/sqrt(nope+rope). One invocation per (b,j,dr).

struct Params {
    bsz: u32,
    n_heads: u32,
    tcols: u32,
    nope: u32,
    rope: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       d_scores: array<f32>;
@group(0) @binding(2) var<storage, read>       q_rot:    array<f32>;
@group(0) @binding(3) var<storage, read_write> d_k_rot:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let T = p.tcols;
    let rp = p.rope;
    let H = p.n_heads;
    let total = p.bsz * T * rp;
    if (gidx >= total) { return; }

    let dr = gidx % rp;
    let r1 = gidx / rp;
    let j = r1 % T;
    let b = r1 / T;
    let scale = inverseSqrt(f32(p.nope + rp));

    var acc = 0.0;
    for (var h: u32 = 0u; h < H; h = h + 1u) {
        let s_hb = (b * H + h) * T;
        for (var i: u32 = j; i < T; i = i + 1u) {
            let s = d_scores[(s_hb + i) * T + j];
            acc = acc + s * q_rot[(b * T + i) * H * rp + h * rp + dr];
        }
    }
    d_k_rot[(b * T + j) * rp + dr] = acc * scale;
}

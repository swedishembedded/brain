// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Banded attention backward, step 3 - gradient w.r.t. q from the band slab
// @how   one thread per output element, serial inner reduction over the band
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Banded attention backward, step 3, the twin of `attn_bwd_dq_cross`:
//   d_q[b,i,h,d] = scale * sum_{s, 0 <= j < t_enc} d_score[b,h,i,s] * k[b,j,h,d],   j = q0 + i + s - window
// Written (assigned) into the q region of the fused grad buffer. One
// invocation per (b,h,i,d).

struct Params {
    bsz: u32,
    n_heads: u32,
    t_dec: u32,
    t_enc: u32,
    head_dim: u32,
    q_stride: u32,
    kv_stride: u32,
    q_off: u32,
    k_off: u32,
    q0: u32,
    window: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       d_scores: array<f32>;
@group(0) @binding(2) var<storage, read>       kv:       array<f32>;
@group(0) @binding(3) var<storage, read_write> d_q:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let W = 2u * p.window + 1u;
    let Tq = p.t_dec;
    let Tk = p.t_enc;
    let hd = p.head_dim;
    let total = p.bsz * p.n_heads * Tq * hd;
    if (idx >= total) { return; }

    let d = idx % hd;
    let r1 = idx / hd;
    let i = r1 % Tq;
    let r2 = r1 / Tq;
    let h = r2 % p.n_heads;
    let b = r2 / p.n_heads;
    let scale = inverseSqrt(f32(hd));

    let s_base = ((b * p.n_heads + h) * Tq + i) * W;
    let j_left = i32(p.q0 + i) - i32(p.window);
    var acc = 0.0;
    for (var s: u32 = 0u; s < W; s = s + 1u) {
        let j = j_left + i32(s);
        if (j < 0 || j >= i32(Tk)) { continue; }
        acc = acc + d_scores[s_base + s] * kv[(b * Tk + u32(j)) * p.kv_stride + p.k_off + h * hd + d];
    }
    d_q[(b * Tq + i) * p.q_stride + p.q_off + h * hd + d] = acc * scale;
}

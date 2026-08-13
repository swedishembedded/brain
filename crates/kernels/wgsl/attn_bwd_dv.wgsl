// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Attention backward, step 2 - gradient w.r.t
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Attention backward, step 2 — gradient w.r.t. v:
//   d_v[b,j,h,d] = sum_{i>=j} probs[b,h,i,j] * d_out[b,i,h,d]
// Written into the v region of d_qkv. One invocation per (b,h,j,d).

struct Params {
    bsz: u32,
    n_heads: u32,
    tcols: u32,
    head_dim: u32,
    qkv_stride: u32,
    v_off: u32,
    d_model: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       probs: array<f32>;
@group(0) @binding(2) var<storage, read>       d_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> d_qkv: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
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

    var acc = 0.0;
    for (var i: u32 = j; i < T; i = i + 1u) {
        let prob = probs[((b * p.n_heads + h) * T + i) * T + j];
        acc = acc + prob * d_out[(b * T + i) * p.d_model + h * hd + d];
    }
    d_qkv[(b * T + j) * p.qkv_stride + p.v_off + h * hd + d] = acc;
}

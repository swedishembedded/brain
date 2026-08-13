// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Attention output: out[b,i,h,d] = sum_{j<=i} probs[b,h,i,j] * v[b,j,h,d]
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Attention output: out[b,i,h,d] = sum_{j<=i} probs[b,h,i,j] * v[b,j,h,d].
// v read from the qkv buffer; out written contiguous [B*T, d_model].
// One invocation per (b,h,i,d).

struct Params {
    bsz: u32,
    n_heads: u32,
    tcols: u32,        // T
    head_dim: u32,
    qkv_stride: u32,   // 3*d_model
    v_off: u32,        // 2*d_model
    d_model: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       probs: array<f32>;
@group(0) @binding(2) var<storage, read>       qkv:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out:   array<f32>;

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
    let i = r1 % T;
    let r2 = r1 / T;
    let h = r2 % p.n_heads;
    let b = r2 / p.n_heads;

    let p_base = ((b * p.n_heads + h) * T + i) * T;
    var acc = 0.0;
    for (var j: u32 = 0u; j <= i; j = j + 1u) {
        let v = qkv[(b * T + j) * p.qkv_stride + p.v_off + h * hd + d];
        acc = acc + probs[p_base + j] * v;
    }
    out[(b * T + i) * p.d_model + h * hd + d] = acc;
}

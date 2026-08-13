// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  GQA attention output, separate v buffer
// @how   one thread per output element
// @opt   3
// @cpu   native
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// GQA attention output, separate v buffer:
//   ctx[b,i,h,d] = sum_{j<=i} probs[b,h,i,j] * v[b,j,hkv,d],  hkv = h / group.
// v is [B*T, n_kv_heads*head_dim]; ctx is [B*T, n_heads*head_dim], head-major.
// One invocation per (b,h,i,d).

struct Params {
    bsz: u32,
    n_heads: u32,
    n_kv_heads: u32,
    tcols: u32,        // T
    head_dim: u32,
    group: u32,        // n_heads / n_kv_heads
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       probs: array<f32>;
@group(0) @binding(2) var<storage, read>       v:     array<f32>;
@group(0) @binding(3) var<storage, read_write> ctx:   array<f32>;

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

    let hkv = h / p.group;
    let k_row = p.n_kv_heads * hd;
    let q_row = p.n_heads * hd;
    let p_base = ((b * p.n_heads + h) * T + i) * T;
    var acc = 0.0;
    for (var j: u32 = 0u; j <= i; j = j + 1u) {
        let vv = v[(b * T + j) * k_row + hkv * hd + d];
        acc = acc + probs[p_base + j] * vv;
    }
    ctx[(b * T + i) * q_row + h * hd + d] = acc;
}

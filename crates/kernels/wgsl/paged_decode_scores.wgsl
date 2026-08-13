// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Paged decode-step attention scores
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Paged decode-step attention scores: a single query vs all `t` cached keys,
// where each key's physical address is resolved through the block table.
//   scores[h,j] = (q[h] . pool_k[block_table[j/bs]*bs + j%bs, kvhead(h)]) * scale
// One invocation per (h, j<t). GQA: kvhead(h)=h/group. Barrier-free.

struct Params {
    n_heads: u32,
    group: u32,       // n_heads / n_kv_heads
    head_dim: u32,
    t: u32,           // cached length
    block_size: u32,
    kv_stride: u32,   // n_kv_heads * head_dim
    cap: u32,         // scores row stride
    scale: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:           array<f32>;
@group(0) @binding(2) var<storage, read>       pool_k:      array<f32>;
@group(0) @binding(3) var<storage, read>       block_table: array<u32>;
@group(0) @binding(4) var<storage, read_write> scores:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.n_heads * p.t) { return; }
    let h = idx / p.t;
    let j = idx % p.t;
    let hd = p.head_dim;
    let kvh = h / p.group;
    let physical = block_table[j / p.block_size];
    let slot = (physical * p.block_size + (j % p.block_size)) * p.kv_stride + kvh * hd;
    let qb = h * hd;
    var s = 0.0;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        s = s + q[qb + d] * pool_k[slot + d];
    }
    scores[h * p.cap + j] = s * p.scale;
}

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Paged decode-step attention apply
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Paged decode-step attention apply: context = probs-weighted sum of the cached
// values, each value addressed through the block table.
//   ctx[h,d] = sum_j probs[h,j] * pool_v[block_table[j/bs]*bs + j%bs, kvhead(h), d]
// One invocation per (h, d). GQA: kvhead(h)=h/group. Barrier-free.

struct Params {
    n_heads: u32,
    group: u32,
    head_dim: u32,
    t: u32,
    block_size: u32,
    kv_stride: u32,
    cap: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       probs:       array<f32>;
@group(0) @binding(2) var<storage, read>       pool_v:      array<f32>;
@group(0) @binding(3) var<storage, read>       block_table: array<u32>;
@group(0) @binding(4) var<storage, read_write> ctx:         array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.n_heads * p.head_dim) { return; }
    let h = idx / p.head_dim;
    let d = idx % p.head_dim;
    let hd = p.head_dim;
    let kvh = h / p.group;
    let pbase = h * p.cap;
    var acc = 0.0;
    for (var j: u32 = 0u; j < p.t; j = j + 1u) {
        let physical = block_table[j / p.block_size];
        let slot = (physical * p.block_size + (j % p.block_size)) * p.kv_stride + kvh * hd + d;
        acc = acc + probs[pbase + j] * pool_v[slot];
    }
    ctx[idx] = acc;
}

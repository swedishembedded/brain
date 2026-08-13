// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Batched paged decode apply
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32|bf16
// @tpl   pool_v -> bf16 storage variant (B9: exact-shift decode, same mechanism
//        as the weight-consuming kernels, applied here to a KV-cache page
//        instead of a static weight; `slot` was already a bare identifier, no
//        hoist needed)
//
// Batched paged decode apply: context per (b,h,d) = probs-weighted sum of cached
// values addressed through sequence b's block table. ctx : [batch, n_heads*head_dim].
struct Params {
    batch: u32,
    n_heads: u32,
    group: u32,
    head_dim: u32,
    block_size: u32,
    kv_stride: u32,
    cap: u32,
    max_bt: u32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       probs:        array<f32>;
@group(0) @binding(2) var<storage, read>       pool_v:       array<f32>;
@group(0) @binding(3) var<storage, read>       block_tables: array<u32>;
@group(0) @binding(4) var<storage, read>       seq_lens:     array<u32>;
@group(0) @binding(5) var<storage, read_write> ctx:          array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.batch * p.n_heads * p.head_dim) { return; }
    let hd = p.head_dim;
    let b = idx / (p.n_heads * hd);
    let rem = idx % (p.n_heads * hd);
    let h = rem / hd;
    let d = rem % hd;
    let t = seq_lens[b];
    let kvh = h / p.group;
    let pbase = (b * p.n_heads + h) * p.cap;
    var acc = 0.0;
    for (var j: u32 = 0u; j < t; j = j + 1u) {
        let physical = block_tables[b * p.max_bt + j / p.block_size];
        let slot = (physical * p.block_size + (j % p.block_size)) * p.kv_stride + kvh * hd + d;
        acc = acc + probs[pbase + j] * pool_v[slot];
    }
    ctx[b * (p.n_heads * hd) + h * hd + d] = acc;
}

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Batched paged decode scores
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32|bf16
// @tpl   pool_k -> bf16 storage variant (B9: exact-shift decode, same mechanism
//        as the weight-consuming kernels, applied here to a KV-cache page
//        instead of a static weight)
//
// Batched paged decode scores: for each sequence b in the batch, its single query
// attends all seq_lens[b] cached keys via that sequence's block table.
//   q      : [batch, n_heads*head_dim]
//   scores : [batch, n_heads, cap]         (only j<seq_lens[b] written)
struct Params {
    batch: u32,
    n_heads: u32,
    group: u32,
    head_dim: u32,
    block_size: u32,
    kv_stride: u32,
    cap: u32,       // scores row stride (>= max seq len)
    max_bt: u32,    // block_tables row stride (blocks per sequence)
    scale: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:            array<f32>;
@group(0) @binding(2) var<storage, read>       pool_k:       array<f32>;
@group(0) @binding(3) var<storage, read>       block_tables: array<u32>;
@group(0) @binding(4) var<storage, read>       seq_lens:     array<u32>;
@group(0) @binding(5) var<storage, read_write> scores:       array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.batch * p.n_heads * p.cap) { return; }
    let b = idx / (p.n_heads * p.cap);
    let rem = idx % (p.n_heads * p.cap);
    let h = rem / p.cap;
    let j = rem % p.cap;
    if (j >= seq_lens[b]) { return; }
    let hd = p.head_dim;
    let kvh = h / p.group;
    let physical = block_tables[b * p.max_bt + j / p.block_size];
    let slot = (physical * p.block_size + (j % p.block_size)) * p.kv_stride + kvh * hd;
    let qb = (b * p.n_heads + h) * hd;
    var s = 0.0;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        // Hoisted to a bare identifier (B9) so `kernels::template::dtype_variant`
        // can template a bf16-packed `pool_k` tier - the decode expansion reads
        // the index twice, so a compound expression would be double-evaluated.
        // Same pattern B4 used for `matmul.wgsl`'s `wi`.
        let ki = slot + d;
        s = s + q[qb + d] * pool_k[ki];
    }
    scores[(b * p.n_heads + h) * p.cap + j] = s * p.scale;
}

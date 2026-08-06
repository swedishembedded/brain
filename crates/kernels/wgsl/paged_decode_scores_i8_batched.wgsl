// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Batched paged decode scores over an INT8 pool
// @how   one thread per output element, serial inner reduction
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant int8
//
// Batched paged decode scores over an INT8 pool: like paged_decode_scores_batched
// but each cached key is dequantised (int8 * per-(token,kv-head) scale) on read.
struct Params {
    batch: u32, n_heads: u32, group: u32, head_dim: u32,
    block_size: u32, kv_stride: u32, cap: u32, max_bt: u32, scale: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       q:            array<f32>;
@group(0) @binding(2) var<storage, read>       pool:         array<u32>;
@group(0) @binding(3) var<storage, read>       block_tables: array<u32>;
@group(0) @binding(4) var<storage, read>       seq_lens:     array<u32>;
@group(0) @binding(5) var<storage, read>       scales:       array<f32>;
@group(0) @binding(6) var<storage, read_write> scores:       array<f32>;

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
    let n_kv = p.kv_stride / hd;
    let kvh = h / p.group;
    let physical = block_tables[b * p.max_bt + j / p.block_size];
    let slot = physical * p.block_size + (j % p.block_size);
    let qscale = scales[slot * n_kv + kvh];
    let base_i8 = slot * p.kv_stride + kvh * hd;
    let qb = (b * p.n_heads + h) * hd;
    var acc = 0.0;
    for (var d: u32 = 0u; d < hd; d = d + 1u) {
        let e = base_i8 + d;
        let byte = (pool[e / 4u] >> (8u * (e % 4u))) & 0xffu;
        let iv = i32(byte);
        let sv = select(iv, iv - 256, iv > 127);
        acc = acc + q[qb + d] * f32(sv);
    }
    scores[(b * p.n_heads + h) * p.cap + j] = acc * qscale * p.scale;
}

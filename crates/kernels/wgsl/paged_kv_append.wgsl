// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Write one token's projected K (or V) into a paged KV block pool at a physical
// block + offset: pool[(block*block_size + offset)*kv_stride + c] = src[c].
// The (block, offset) come from the sequence's block table, computed host-side
// for the new token. One invocation per element. Distinct src/pool (no alias).

struct Params {
    kv_stride: u32,   // n_kv_heads * head_dim
    block: u32,       // physical block id
    offset: u32,      // token slot within the block
    block_size: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       src:  array<f32>;
@group(0) @binding(2) var<storage, read_write> pool: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.kv_stride) { return; }
    pool[(p.block * p.block_size + p.offset) * p.kv_stride + idx] = src[idx];
}

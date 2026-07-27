// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Append a batch of new tokens' K (or V) into the paged pool: sequence b writes
// src[b, :] into pool at its per-sequence (blocks[b], offsets[b]).
struct Params { batch: u32, kv_stride: u32, block_size: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       src:     array<f32>;
@group(0) @binding(2) var<storage, read>       blocks:  array<u32>;
@group(0) @binding(3) var<storage, read>       offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> pool:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.batch * p.kv_stride) { return; }
    let b = idx / p.kv_stride;
    let c = idx % p.kv_stride;
    pool[(blocks[b] * p.block_size + offsets[b]) * p.kv_stride + c] = src[b * p.kv_stride + c];
}

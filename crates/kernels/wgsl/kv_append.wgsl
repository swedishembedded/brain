// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Append one token's projected K (or V) into a KV cache: write the length-`width`
// source row into cache row `row`, i.e. dst[row*width + i] = src[i]. `dst` is the
// persistent [max_T, width] cache buffer; `src` is the freshly projected (and, for
// K, already QK-normed + RoPE'd) new-token vector. One invocation per element.
// Barrier-free; `src` and `dst` are distinct buffers (no output alias).

struct Params {
    width: u32,   // kv_dim = n_kv_heads * head_dim
    row: u32,     // cache row to write (the token's absolute position)
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.width) { return; }
    dst[p.row * p.width + idx] = src[idx];
}

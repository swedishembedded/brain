// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Residual splice (forward): copy a compact `[n]` source block into `dst`
// starting at flat element offset `base`:
//   dst[base + i] = src[i]   for i in 0..n.
// Used by the vision-language models to write projected image-token embeddings
// over a contiguous row range of the decoder residual stream (res[0]) after the
// text token-embedding gather. `base = row0 * d_model`, `n = n_img_rows *
// d_model`; call once per contiguous image run. One invocation per element.
// (Distinct from region_copy, which reads and writes the SAME index — here the
// source is compact and the destination offset is an independent parameter, so
// there is no storage-binding alignment constraint.)

struct Params {
    n:    u32,
    base: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.n) { return; }
    dst[p.base + idx] = src[idx];
}

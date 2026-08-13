// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Add learned absolute positional embeddings in place
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// Add learned absolute positional embeddings in place:
//   x[row, c] += pos[(row % T) * D + c]
// row = global token index within the [B*T, D] batch; position = row % T.
// One invocation per element (total = B*T*D).

struct Params {
    total: u32,
    d_model: u32,
    t: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> x:   array<f32>;
@group(0) @binding(2) var<storage, read>       pos: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    if (idx >= p.total) { return; }
    let c = idx % p.d_model;
    let row = idx / p.d_model;
    let pos_row = row % p.t;
    x[idx] = x[idx] + pos[pos_row * p.d_model + c];
}

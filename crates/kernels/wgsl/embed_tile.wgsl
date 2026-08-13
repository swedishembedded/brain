// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Embedding gather over a VOCAB TILE
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// Embedding gather over a VOCAB TILE: x[t,c] = emb[token[t], c], but `emb` is
// bound to a sub-range covering rows [v0, v0+v_count) of the full table, so a
// >128MB embedding can be gathered in several passes each within the binding
// size limit. An invocation writes its output element only when its token falls
// in this tile's range; every token belongs to exactly one tile, so across all
// tiles every element is written exactly once. One invocation per (t, c).

struct Params {
    d_model: u32,
    seq_len: u32,
    v0: u32,       // first vocab row in this tile (absolute)
    v_count: u32,  // rows in this tile
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       tokens: array<u32>;
@group(0) @binding(2) var<storage, read>       emb:    array<f32>;  // tile rows only
@group(0) @binding(3) var<storage, read_write> x:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let total = p.seq_len * p.d_model;
    if (idx >= total) { return; }
    let t = idx / p.d_model;
    let c = idx % p.d_model;
    let tok = tokens[t];
    if (tok < p.v0 || tok >= p.v0 + p.v_count) { return; }
    x[idx] = emb[(tok - p.v0) * p.d_model + c];
}

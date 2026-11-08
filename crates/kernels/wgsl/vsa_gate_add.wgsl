// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Video sparse attention: add the coarse branch to the attention output, scaled per row and channel by the gate
// @how   one thread per output element, the row's tile read from a row-to-tile map
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// out[r, h * head_dim + c] += coarse[h, tile[r], c] * gate[r, h * head_dim + c]
//
// `out` and `gate` are `[rows, heads * head_dim]` in packed order: the fine
// attention's output and the block's gate projection of its input. `tile`
// maps each packed row to its tile.

struct Params {
    rows: u32,
    tiles: u32,
    heads: u32,
    head_dim: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> out:    array<f32>;
@group(0) @binding(2) var<storage, read>       coarse: array<f32>;
@group(0) @binding(3) var<storage, read>       gate:   array<f32>;
@group(0) @binding(4) var<storage, read>       tile:   array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let width = p.heads * p.head_dim;
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.rows * width) { return; }
    let r = idx / width;
    let col = idx - r * width;
    let h = col / p.head_dim;
    let c = col - h * p.head_dim;
    out[idx] = out[idx] + coarse[(h * p.tiles + tile[r]) * p.head_dim + c] * gate[idx];
}

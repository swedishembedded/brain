// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Video sparse attention: the coarse branch, softmax of a query tile's scores over every key tile times the tiles' mean v
// @how   one thread per (head, query tile, channel), a max pass and a weighted-sum pass over the tiles
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// out[h, i, c] = sum_j softmax_j(s[h, i, .]) * vm[h, j, c]
//
// `vm` is `vsa_tile_mean` of v. The branch reaches every tile at tile
// resolution; the block adds it to the fine attention scaled by its gate
// (`vsa_gate_add`).

struct Params {
    tiles: u32,
    heads: u32,
    head_dim: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read>       s:   array<f32>;
@group(0) @binding(3) var<storage, read>       vm:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.heads * p.tiles * p.head_dim) { return; }
    let c = idx % p.head_dim;
    let i = (idx / p.head_dim) % p.tiles;
    let h = idx / (p.head_dim * p.tiles);
    let row = (h * p.tiles + i) * p.tiles;
    var mx = -3.4e38;
    for (var j = 0u; j < p.tiles; j = j + 1u) {
        mx = max(mx, s[row + j]);
    }
    var total = 0.0;
    var acc = 0.0;
    for (var j = 0u; j < p.tiles; j = j + 1u) {
        let w = exp(s[row + j] - mx);
        total = total + w;
        acc = acc + w * vm[(h * p.tiles + j) * p.head_dim + c];
    }
    out[idx] = acc / total;
}

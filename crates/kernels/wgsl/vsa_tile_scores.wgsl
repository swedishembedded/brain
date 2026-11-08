// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Video sparse attention: every query tile's score of every key tile, mean q . mean k scaled
// @how   one thread per (head, query tile, key tile), serial dot product over head_dim
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// s[h, i, j] = scale * sum_c qm[h, i, c] * km[h, j, c]
//
// `qm`/`km` are `vsa_tile_mean`'s `[heads, tiles, head_dim]` tile means. The
// score matrix is `heads * tiles^2` - a few hundred tiles per head for a clip
// - so it is small next to the attention it steers; consecutive threads take
// consecutive key tiles and share the query tile's row through the cache.

struct Params {
    tiles: u32,
    heads: u32,
    head_dim: u32,
    scale: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> s:  array<f32>;
@group(0) @binding(2) var<storage, read>       qm: array<f32>;
@group(0) @binding(3) var<storage, read>       km: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.heads * p.tiles * p.tiles) { return; }
    let j = idx % p.tiles;
    let i = (idx / p.tiles) % p.tiles;
    let h = idx / (p.tiles * p.tiles);
    let qo = (h * p.tiles + i) * p.head_dim;
    let ko = (h * p.tiles + j) * p.head_dim;
    var acc = 0.0;
    for (var c = 0u; c < p.head_dim; c = c + 1u) {
        acc = acc + qm[qo + c] * km[ko + c];
    }
    s[idx] = acc * p.scale;
}

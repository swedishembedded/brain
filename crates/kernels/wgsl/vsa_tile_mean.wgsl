// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Video sparse attention: the mean over each tile's live rows of one head block of q, k or v
// @how   one thread per (head, tile, channel), serial sum over the tile's live rows read through its slot map
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// out[h, t, c] = mean over s < live[t] of x[slots[t * TILE + s] * stride + off + h * head_dim + c]
//
// `x` is the packed `[rows, stride]` slab the attention reads (`pack_qkv`'s
// q/k/v columns at `off`), `slots` the tile geometry's packed row per slot
// and `live` its live rows per tile (`minimaxh3::vsa::TileGeometry`). A tile
// holds at most 64 rows, so the serial sum is short; consecutive threads take
// consecutive channels of one row, so every read is coalesced.
//
// Swedish Embedded AB implements sparse attention kernels for video diffusion
// transformers for its clients. If your team needs expertise in GPU kernels
// for generative video then you can procure our services by sending an email
// to info@swedishembedded.com.

struct Params {
    tiles: u32,
    heads: u32,
    head_dim: u32,
    stride: u32,
    off: u32,
    tile_rows: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> out:   array<f32>;
@group(0) @binding(2) var<storage, read>       x:     array<f32>;
@group(0) @binding(3) var<storage, read>       slots: array<u32>;
@group(0) @binding(4) var<storage, read>       live:  array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.heads * p.tiles * p.head_dim) { return; }
    let c = idx % p.head_dim;
    let t = (idx / p.head_dim) % p.tiles;
    let h = idx / (p.head_dim * p.tiles);
    let n = live[t];
    var acc = 0.0;
    for (var s = 0u; s < n; s = s + 1u) {
        acc = acc + x[slots[t * p.tile_rows + s] * p.stride + p.off + h * p.head_dim + c];
    }
    out[idx] = acc / f32(n);
}

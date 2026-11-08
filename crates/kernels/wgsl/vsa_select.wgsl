// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Video sparse attention: which key tiles each query tile attends, a per-region top-k of tile scores
// @how   one thread per (head, query tile, key tile): kept when either is a prefix tile, else when the key's rank within its video region is below the region's keep
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// mask[h, i, j] = 1 when query tile i or key tile j is one of the leading
// `prefix` tiles, else when key tile j's rank among its video region's tiles
// by s[h, i, .] - higher first, ties to the lower tile - is below that
// region's keep. `spans` holds `(start, end, keep)` per region.
//
// Deciding each element from its own rank needs no sort and no compaction:
// every element is independent, so the mask is one dispatch. The rank is a
// count over the region (a few hundred tiles), against an attention that
// reads every kept tile's 64 rows. It is the same set a stable top-k picks.

struct Params {
    tiles: u32,
    heads: u32,
    prefix: u32,
    regions: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> mask:  array<u32>;
@group(0) @binding(2) var<storage, read>       s:     array<f32>;
@group(0) @binding(3) var<storage, read>       spans: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.heads * p.tiles * p.tiles) { return; }
    let j = idx % p.tiles;
    let i = (idx / p.tiles) % p.tiles;
    if (i < p.prefix || j < p.prefix) {
        mask[idx] = 1u;
        return;
    }
    let row = idx - j;
    let sj = s[idx];
    var keep = 0u;
    for (var r = 0u; r < p.regions; r = r + 1u) {
        let start = spans[3u * r];
        let end = spans[3u * r + 1u];
        if (j >= start && j < end) {
            var rank = 0u;
            for (var l = start; l < end; l = l + 1u) {
                let sl = s[row + l];
                if (sl > sj || (sl == sj && l < j)) { rank = rank + 1u; }
            }
            keep = select(0u, 1u, rank < spans[3u * r + 2u]);
        }
    }
    mask[idx] = keep;
}

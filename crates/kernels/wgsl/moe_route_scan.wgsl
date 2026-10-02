// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Sparse-MoE routing tables, stage 2: exclusive offsets of the per-expert slot and tile counts
// @how   one thread walks the experts (a few hundred) serially
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// See `moe_route_count.wgsl`. Writes the table `moe_i8_grouped.wgsl` and
// `moe_route_emit.wgsl` read, `2 * (ne + 1)` words:
//
//   tab[e]            first position of expert e in the expert-major order
//                     (`tab[ne]` = total slots)
//   tab[ne + 1 + e]   first MR-slot tile of expert e (`tab[2 * ne + 1]` = total
//                     tiles), a tile being `mr` consecutive slots of one expert
//
// An expert with no slots owns no tile. `ne` is a few hundred, so one thread is
// cheaper than a parallel scan's extra launches.
//
//   params : u32 [ne, mr]
//   counts : [ne] u32
//   tab    : [2 * (ne + 1)] u32

struct Params {
    ne: u32,
    mr: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> tab:    array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // One thread of one workgroup (`Workgroups(1)`).
    if (gid.y * (nwg.x * 64u) + gid.x != 0u) { return; }
    var pos = 0u;
    var tiles = 0u;
    for (var e: u32 = 0u; e < p.ne; e = e + 1u) {
        tab[e] = pos;
        tab[p.ne + 1u + e] = tiles;
        let c = counts[e];
        pos = pos + c;
        tiles = tiles + (c + p.mr - 1u) / p.mr;
    }
    tab[p.ne] = pos;
    tab[2u * p.ne + 1u] = tiles;
}

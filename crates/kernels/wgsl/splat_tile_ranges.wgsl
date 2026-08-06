// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Tiled 3DGS, stage 4: per-tile [start, end) ranges over the SORTED keys, by neighbor comparison (disjoint writes, no atomics)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// Tiled 3DGS, stage 4: per-tile [start, end) ranges over the SORTED keys, by
// neighbor comparison (disjoint writes, no atomics). ranges must be
// zero-cleared before this dispatch so empty tiles read start == end == 0.

struct Params {
    n_isects: u32,
    depth_bits: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       keys:   array<u32>;
@group(0) @binding(2) var<storage, read_write> ranges: array<u32>; // n_tiles*2

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.n_isects) { return; }
    let t = keys[i] >> p.depth_bits;
    if (i == 0u || t != (keys[i - 1u] >> p.depth_bits)) {
        ranges[t * 2u] = i;
    }
    if (i + 1u == p.n_isects || t != (keys[i + 1u] >> p.depth_bits)) {
        ranges[t * 2u + 1u] = i + 1u;
    }
}

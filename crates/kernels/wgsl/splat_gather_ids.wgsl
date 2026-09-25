// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Tiled 3DGS, after the sort: the gaussian id of every sorted instance
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// `splat_emit.wgsl` sorts each instance's emission index rather than its
// gaussian id, so the backward knows where to put the instance's gradient
// record. The rasterizers want the gaussian: out[j] = ids[order[j]].

struct Params {
    n: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       order: array<u32>; // sorted emission indices
@group(0) @binding(2) var<storage, read>       ids:   array<u32>; // emission index -> gaussian
@group(0) @binding(3) var<storage, read_write> out:   array<u32>; // sorted gaussian ids

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let j = gid.y * (nwg.x * 64u) + gid.x;
    if (j >= p.n) { return; }
    out[j] = ids[order[j]];
}

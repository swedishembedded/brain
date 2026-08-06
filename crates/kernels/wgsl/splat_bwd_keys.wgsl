// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  3DGS backward, stage 3 prep
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// 3DGS backward, stage 3 prep: extract the gaussian-id sort keys from the
// gradient records (slot 9 holds the id bit-cast to f32) and index payloads.

struct Params {
    n: u32, // record count
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       recs: array<f32>; // n*10
@group(0) @binding(2) var<storage, read_write> keys: array<u32>;
@group(0) @binding(3) var<storage, read_write> vals: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.n) { return; }
    keys[i] = bitcast<u32>(recs[i * 10u + 9u]);
    vals[i] = i;
}

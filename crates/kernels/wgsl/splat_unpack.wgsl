// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Unpack the fit-time packed gaussian geometry [N*10] = {mean(3), scale(3), quat(4)} into the separate forward-kernel buffers
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Unpack the fit-time packed gaussian geometry [N*10] = {mean(3), scale(3),
// quat(4)} into the separate forward-kernel buffers. One invocation per
// gaussian. (Packed layout matches d_gauss so AdamW runs on one buffer.)

struct Params {
    n: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       packed: array<f32>;
@group(0) @binding(2) var<storage, read_write> means:  array<f32>;
@group(0) @binding(3) var<storage, read_write> scales: array<f32>;
@group(0) @binding(4) var<storage, read_write> quats:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.n) { return; }
    let o = i * 10u;
    means[i * 3u] = packed[o];
    means[i * 3u + 1u] = packed[o + 1u];
    means[i * 3u + 2u] = packed[o + 2u];
    scales[i * 3u] = packed[o + 3u];
    scales[i * 3u + 1u] = packed[o + 4u];
    scales[i * 3u + 2u] = packed[o + 5u];
    quats[i * 4u] = packed[o + 6u];
    quats[i * 4u + 1u] = packed[o + 7u];
    quats[i * 4u + 2u] = packed[o + 8u];
    quats[i * 4u + 3u] = packed[o + 9u];
}

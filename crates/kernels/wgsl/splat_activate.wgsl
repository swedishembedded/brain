// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  splat parameters to renderer inputs: exp(log scale), sigmoid(logit)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// A fit keeps each gaussian as 11 raw parameters, {mean (3), log scale (3),
// quaternion (4), opacity logit} (`splat_adam.wgsl`); the renderer draws
// activated ones. One invocation per gaussian writes the renderer's four
// inputs: mean and quaternion as they are, scale = exp(log scale), opacity =
// sigmoid(logit).

struct Params {
    n: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       geo:    array<f32>; // N*11
@group(0) @binding(2) var<storage, read_write> means:  array<f32>; // N*3
@group(0) @binding(3) var<storage, read_write> scales: array<f32>; // N*3
@group(0) @binding(4) var<storage, read_write> quats:  array<f32>; // N*4
@group(0) @binding(5) var<storage, read_write> opac:   array<f32>; // N

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.n) { return; }
    let o = i * 11u;
    means[i * 3u] = geo[o];
    means[i * 3u + 1u] = geo[o + 1u];
    means[i * 3u + 2u] = geo[o + 2u];
    scales[i * 3u] = exp(geo[o + 3u]);
    scales[i * 3u + 1u] = exp(geo[o + 4u]);
    scales[i * 3u + 2u] = exp(geo[o + 5u]);
    quats[i * 4u] = geo[o + 6u];
    quats[i * 4u + 1u] = geo[o + 7u];
    quats[i * 4u + 2u] = geo[o + 8u];
    quats[i * 4u + 3u] = geo[o + 9u];
    opac[i] = 1.0 / (1.0 + exp(-geo[o + 10u]));
}

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  ReLU6: min(max(x, 0), 6)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// ReLU6 (Howard et al., "MobileNets", 2017): ReLU capped at 6, torch's
// `F.relu6`, used by MobileNetV2-style inverted-residual heads.
// Elementwise over `total`; the backward is relu6_bwd.wgsl.

struct Params {
    total: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let v = x[idx];
    out[idx] = clamp(v, 0.0, 6.0);
}

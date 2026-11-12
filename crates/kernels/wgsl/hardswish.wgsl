// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Hard swish, MobileNetV3's activation: x * relu6(x + 3) / 6
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// Hard swish (Howard et al., "Searching for MobileNetV3", 2019): the
// piecewise-polynomial stand-in for SiLU,
//   hardswish(x) = x * min(max(x + 3, 0), 6) / 6
// zero below -3, the identity above +3, x(x + 3)/6 between. Matches torch's
// `F.hardswish`, which is what MobileNetV3 checkpoints were trained with.
// Elementwise over `total`; the backward is hardswish_bwd.wgsl.

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
    out[idx] = v * clamp(v + 3.0, 0.0, 6.0) / 6.0;
}

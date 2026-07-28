// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Moondream MoE expert activation — GeGLU with a +1 shift:
//   out[i] = gelu_erf(h[i]) * (g[i] + 1)
// `h` and `g` are the two halves of the expert's fc1 projection. erf-GELU matches
// torch's default F.gelu (A&S 7.1.26 erf, inlined + branch-based sign so the
// wgsl-cpu JIT accepts it, like gelu_erf.wgsl). Elementwise over `total`.

struct Params {
    total: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       h:   array<f32>;
@group(0) @binding(2) var<storage, read>       g:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.total) { return; }
    let v = h[idx];
    let arg = v * 0.7071067811865476;
    var s = 1.0;
    if (arg < 0.0) { s = -1.0; }
    let ax = abs(arg);
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    let erf = s * (1.0 - poly * exp(-ax * ax));
    let gh = 0.5 * v * (1.0 + erf);
    out[idx] = gh * (g[idx] + 1.0);
}

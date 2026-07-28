// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// geglu_shift backward w.r.t. `h`:
//   dh[i] = dy[i] * (g[i] + 1) * gelu'(h[i])
// gelu'(x) = Phi(x) + x·phi(x) = 0.5·(1+erf(x/√2)) + x·(1/√(2π))·exp(-x²/2).
// erf inlined (A&S), no helper fn (wgsl-cpu JIT). Inputs dy, g, h; output dh.

struct Params {
    total: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read>       g:  array<f32>;
@group(0) @binding(3) var<storage, read>       h:  array<f32>;
@group(0) @binding(4) var<storage, read_write> dh: array<f32>;

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
    let phi = 0.5 * (1.0 + erf);                       // Phi(x)
    let pdf = 0.3989422804014327 * exp(-0.5 * v * v);  // N(0,1) pdf
    let dgelu = phi + v * pdf;
    dh[idx] = dy[idx] * (g[idx] + 1.0) * dgelu;
}

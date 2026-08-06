// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Exact (erf-based) GELU, matching torch's default `F.gelu`
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// Exact (erf-based) GELU, matching torch's default `F.gelu`:
//   gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
// brain's `gelu` uses the tanh approximation (GPT-2 style); GenieRedux and
// other torch models that call plain F.gelu need this exact form for parity.
// erf via Abramowitz & Stegun 7.1.26 (max abs error ~1.5e-7, well under fp32).
// Inlined (no helper fn) and branch-based sign so the wgsl-cpu JIT accepts it.
// Elementwise over `total`.

struct Params {
    total: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    if (idx >= p.total) { return; }
    let v = x[idx];
    let arg = v * 0.7071067811865476;
    var s = 1.0;
    if (arg < 0.0) { s = -1.0; }
    let ax = abs(arg);
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    let erf = s * (1.0 - poly * exp(-ax * ax));
    out[idx] = 0.5 * v * (1.0 + erf);
}

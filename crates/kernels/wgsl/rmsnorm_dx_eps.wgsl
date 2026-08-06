// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  RMSNorm backward w.r.t. x, with a RUNTIME epsilon (eps-parameterized twin of rmsnorm_dx, which hardcodes 1e-6)
// @how   one thread per output element, 3 nested serial reductions
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// RMSNorm backward w.r.t. x, with a RUNTIME epsilon (eps-parameterized twin of
// rmsnorm_dx, which hardcodes 1e-6). Forward: y_c = w_c·x_c·r, r=1/sqrt(mean(x²)+eps).
// dX_i = r·w_i·dY_i − (r³·x_i/d)·Σ_c dY_c·w_c·x_c. Z-Image uses eps=1e-5, so the
// backward must recompute r with the same eps as the forward. One row per invocation.

struct Params {
    d_model: u32,
    n_rows: u32,
    eps: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:      array<f32>;
@group(0) @binding(2) var<storage, read>       weight: array<f32>;
@group(0) @binding(3) var<storage, read>       dy:     array<f32>;
@group(0) @binding(4) var<storage, read_write> dx:     array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let n = gidx;
    if (n >= p.n_rows) { return; }
    let d = p.d_model;
    let base = n * d;
    var ss = 0.0;
    for (var c: u32 = 0u; c < d; c = c + 1u) {
        let v = x[base + c];
        ss = ss + v * v;
    }
    let r = inverseSqrt(ss / f32(d) + p.eps);
    var a = 0.0;
    for (var c: u32 = 0u; c < d; c = c + 1u) {
        a = a + dy[base + c] * weight[c] * x[base + c];
    }
    let coef = r * r * r * a / f32(d);
    for (var c: u32 = 0u; c < d; c = c + 1u) {
        dx[base + c] = r * weight[c] * dy[base + c] - coef * x[base + c];
    }
}

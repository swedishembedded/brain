// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  GELU backward (tanh approximation) - gradient w.r.t
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// GELU backward (tanh approximation) — gradient w.r.t. the pre-activation `x`.
//   g(x)  = 0.5 * x * (1 + t),      t = tanh(u),  u = k*(x + 0.044715 x^3)
//   g'(x) = 0.5*(1 + t) + 0.5*x*(1 - t^2) * u',   u' = k*(1 + 3*0.044715 x^2)
//   dx[i] = dout[i] * g'(x[i])
// Elementwise; must stay consistent with gelu.wgsl for the gradient check.

struct Params {
    total: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;   // pre-activation
@group(0) @binding(2) var<storage, read>       dout: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    if (idx >= p.total) { return; }
    let v = x[idx];
    let k = 0.7978845608028654;          // sqrt(2/pi)
    let inner = k * (v + 0.044715 * v * v * v);
    let t = tanh(inner);
    let dinner = k * (1.0 + 3.0 * 0.044715 * v * v);
    let dgelu = 0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * dinner;
    dx[idx] = dout[idx] * dgelu;
}

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// QuickGELU backward — gradient w.r.t. the pre-activation `x`.
// The matching forward is quick_gelu.wgsl (OpenAI CLIP's sigmoid approximation).
// `gelu_bwd.wgsl` (tanh approximation) and `gelu_erf_bwd.wgsl` (exact erf) are
// the derivatives of DIFFERENT functions and are NOT interchangeable with this
// one — the three GELU forms differ by up to ~1e-2, comfortably inside
// gradcheck's 8% RTOL, so a mispaired backward leaves the gate green while
// training on the gradient of another function (the trap gelu_erf_bwd's header
// documents).
//
//   g(x)  = x * s,                s = sigmoid(1.702 * x)
//   g'(x) = s + 1.702 * x * s * (1 - s)
//   dx[i] = dout[i] * g'(x[i])
//
// The sigmoid is written as `1/(1+exp(-1.702*x))`, the identical expression
// quick_gelu.wgsl uses, so forward and backward cannot drift.
//
// Elementwise over `total`; no reduction, hence no cooperative twin and no
// `DeviceCaps::workgroup_reductions` gate. Bindings/Params mirror
// gelu_erf_bwd.wgsl exactly.

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
    let s = 1.0 / (1.0 + exp(-1.702 * v));
    dx[idx] = dout[idx] * (s + 1.702 * v * s * (1.0 - s));
}

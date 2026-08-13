// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Exact (erf-based) GELU backward - gradient w.r.t
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// Exact (erf-based) GELU backward — gradient w.r.t. the pre-activation `x`.
// The matching forward is gelu_erf.wgsl; `gelu_bwd.wgsl` is the derivative of the
// TANH APPROXIMATION and is NOT interchangeable with this one.
//
//   g(x)  = 0.5 * x * (1 + erf(x / sqrt(2)))
//   g'(x) = 0.5 * (1 + erf(x / sqrt(2))) + x * phi(x),   phi(x) = exp(-x^2/2)/sqrt(2*pi)
//   dx[i] = dout[i] * g'(x[i])
//
// Derivation of the second term: d/dx erf(x/sqrt(2)) = sqrt(2/pi) * exp(-x^2/2),
// so 0.5 * x * sqrt(2/pi) * exp(-x^2/2) = x * exp(-x^2/2) / sqrt(2*pi).
//
// Why this kernel exists at all: tanh-GELU and erf-GELU agree to ~1e-3, and their
// derivatives likewise — comfortably INSIDE gradcheck's 8% RTOL. So pairing
// gelu_erf forward with gelu_bwd backward would leave the gate GREEN while
// training on the gradient of a different function. The mismatch is asserted
// explicitly in gradcheck/tests/glue.rs so that trap stays documented.
//
// erf via Abramowitz & Stegun 7.1.26 (max abs error ~1.5e-7, well under fp32),
// inlined with a branch-based sign to match gelu_erf.wgsl exactly — the forward
// and backward MUST use the same erf approximation or they disagree by more than
// the approximation error. Elementwise over `total`.

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

    // erf(v / sqrt(2)) — identical expression to gelu_erf.wgsl.
    let arg = v * 0.7071067811865476;
    var s = 1.0;
    if (arg < 0.0) { s = -1.0; }
    let ax = abs(arg);
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    let erf = s * (1.0 - poly * exp(-ax * ax));

    // phi(v) = exp(-v^2/2) / sqrt(2*pi);  1/sqrt(2*pi) = 0.3989422804014327
    let phi = 0.3989422804014327 * exp(-0.5 * v * v);
    let dgelu = 0.5 * (1.0 + erf) + v * phi;
    dx[idx] = dout[idx] * dgelu;
}

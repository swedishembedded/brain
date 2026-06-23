// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// RMSNorm backward w.r.t. the input x. Forward: y_c = w_c * x_c * r,
// r = 1/sqrt(mean(x^2)+eps). With A = sum_c dY_c * w_c * x_c,
//   dX_i = r*w_i*dY_i - (r^3 * x_i / d) * A
// One invocation per row (recomputes r from x, so no inv buffer is needed —
// keeps this kernel at 4 storage bindings).

struct Params {
    d_model: u32,
    n_rows: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:      array<f32>;
@group(0) @binding(2) var<storage, read>       weight: array<f32>;
@group(0) @binding(3) var<storage, read>       dy:     array<f32>;
@group(0) @binding(4) var<storage, read_write> dx:     array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= p.n_rows) { return; }
    let d = p.d_model;
    let base = n * d;

    var ss = 0.0;
    for (var c: u32 = 0u; c < d; c = c + 1u) {
        let v = x[base + c];
        ss = ss + v * v;
    }
    let r = inverseSqrt(ss / f32(d) + 1e-6);

    var a = 0.0;
    for (var c: u32 = 0u; c < d; c = c + 1u) {
        a = a + dy[base + c] * weight[c] * x[base + c];
    }
    let coef = r * r * r * a / f32(d);
    for (var c: u32 = 0u; c < d; c = c + 1u) {
        dx[base + c] = r * weight[c] * dy[base + c] - coef * x[base + c];
    }
}

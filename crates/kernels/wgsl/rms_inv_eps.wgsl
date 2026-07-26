// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Per-row inverse RMS with a RUNTIME epsilon: inv[n] = 1/sqrt(mean_c(x²)+eps).
// The eps-parameterized twin of rms_inv (which hardcodes 1e-6) — Z-Image's
// RMSNorm uses eps=1e-5, and the backward must match the forward's eps exactly
// or the gain/x grads drift. One invocation per row. Feeds rmsnorm_dw.

struct Params {
    d_model: u32,
    n_rows: u32,
    eps: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read_write> inv: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let n = gidx;
    if (n >= p.n_rows) { return; }
    let base = n * p.d_model;
    var ss = 0.0;
    for (var c: u32 = 0u; c < p.d_model; c = c + 1u) {
        let v = x[base + c];
        ss = ss + v * v;
    }
    inv[n] = inverseSqrt(ss / f32(p.d_model) + p.eps);
}

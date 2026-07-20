// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// LayerNorm helper: per-row mean and inverse-std.
//   mean[n] = mean_c(x);  inv[n] = 1/sqrt(var+eps),  var = mean_c((x-mean)^2), eps a param
// One invocation per row. Feeds layernorm_dgamma (mirrors rms_inv -> rmsnorm_dw).

struct Params {
    d_model: u32,
    n_rows: u32,
    eps: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:    array<f32>;
@group(0) @binding(2) var<storage, read_write> mean: array<f32>;
@group(0) @binding(3) var<storage, read_write> inv:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let n = gidx;
    if (n >= p.n_rows) { return; }
    let d = p.d_model;
    let base = n * d;
    var m = 0.0;
    for (var c: u32 = 0u; c < d; c = c + 1u) { m = m + x[base + c]; }
    m = m / f32(d);
    var va = 0.0;
    for (var c: u32 = 0u; c < d; c = c + 1u) {
        let dx = x[base + c] - m;
        va = va + dx * dx;
    }
    mean[n] = m;
    inv[n] = inverseSqrt(va / f32(d) + p.eps);
}

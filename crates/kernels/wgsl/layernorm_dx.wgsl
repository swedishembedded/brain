// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// LayerNorm backward w.r.t. x. With xhat=(x-mean)*inv, g=dy*gamma:
//   dx[c] = inv * ( g[c] - mean_k(g) - xhat[c] * mean_k(g*xhat) )
// One invocation per row; recomputes mean/inv from x (keeps to 4 bindings).

struct Params {
    d_model: u32,
    n_rows: u32,
    eps: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:     array<f32>;
@group(0) @binding(2) var<storage, read>       gamma: array<f32>;
@group(0) @binding(3) var<storage, read>       dy:    array<f32>;
@group(0) @binding(4) var<storage, read_write> dx:    array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let n = gidx;
    if (n >= p.n_rows) { return; }
    let d = p.d_model;
    let base = n * d;
    let df = f32(d);

    var mean = 0.0;
    for (var c: u32 = 0u; c < d; c = c + 1u) { mean = mean + x[base + c]; }
    mean = mean / df;
    var va = 0.0;
    for (var c: u32 = 0u; c < d; c = c + 1u) {
        let v = x[base + c] - mean;
        va = va + v * v;
    }
    let inv = inverseSqrt(va / df + p.eps);

    var sum_g = 0.0;
    var sum_gx = 0.0;
    for (var c: u32 = 0u; c < d; c = c + 1u) {
        let g = dy[base + c] * gamma[c];
        let xhat = (x[base + c] - mean) * inv;
        sum_g = sum_g + g;
        sum_gx = sum_gx + g * xhat;
    }
    let mg = sum_g / df;
    let mgx = sum_gx / df;
    for (var c: u32 = 0u; c < d; c = c + 1u) {
        let g = dy[base + c] * gamma[c];
        let xhat = (x[base + c] - mean) * inv;
        dx[base + c] = inv * (g - mg - xhat * mgx);
    }
}

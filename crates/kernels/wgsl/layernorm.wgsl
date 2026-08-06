// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  LayerNorm forward (matches torch.nn.LayerNorm)
// @how   one thread per output element, 3 nested serial reductions
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
//
// LayerNorm forward (matches torch.nn.LayerNorm):
//   mean = mean_c(x);  var = mean_c((x-mean)^2)   (biased/population variance)
//   y[c] = (x[c]-mean) / sqrt(var+eps) * gamma[c] + beta[c],   eps a param (1e-5 torch default)
// One invocation per row (d_model small => per-row loop is fine).

struct Params {
    d_model: u32,
    n_rows: u32,
    eps: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:     array<f32>;
@group(0) @binding(2) var<storage, read>       gamma: array<f32>;
@group(0) @binding(3) var<storage, read>       beta:  array<f32>;
@group(0) @binding(4) var<storage, read_write> out:   array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let n = gidx;
    if (n >= p.n_rows) { return; }
    let d = p.d_model;
    let base = n * d;
    var mean = 0.0;
    for (var c: u32 = 0u; c < d; c = c + 1u) { mean = mean + x[base + c]; }
    mean = mean / f32(d);
    var va = 0.0;
    for (var c: u32 = 0u; c < d; c = c + 1u) {
        let dx = x[base + c] - mean;
        va = va + dx * dx;
    }
    va = va / f32(d);
    let inv = inverseSqrt(va + p.eps);
    for (var c: u32 = 0u; c < d; c = c + 1u) {
        out[base + c] = (x[base + c] - mean) * inv * gamma[c] + beta[c];
    }
}

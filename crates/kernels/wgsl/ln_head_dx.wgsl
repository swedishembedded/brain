// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Strided per-head LayerNorm backward (input grad), the ln_head companion.
// x is the CACHED pre-norm buffer (ln_head normalizes in place, so training
// paths copy the region first); dy/dx use the same strided region layout.
//   x̂ = (x-μ)/σ;  g = dy·γ
//   dx = (g - mean(g) - x̂·mean(g·x̂)) / σ
// One invocation per (row, head).

struct Params {
    rows: u32,
    heads: u32,
    head_dim: u32,
    row_stride: u32,
    off: u32,
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
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.rows * p.heads) { return; }
    let h = idx % p.heads;
    let row = idx / p.heads;
    let hd = p.head_dim;
    let base = row * p.row_stride + p.off + h * hd;
    var mean = 0.0;
    for (var c = 0u; c < hd; c = c + 1u) { mean = mean + x[base + c]; }
    mean = mean / f32(hd);
    var va = 0.0;
    for (var c = 0u; c < hd; c = c + 1u) {
        let d = x[base + c] - mean;
        va = va + d * d;
    }
    let inv = inverseSqrt(va / f32(hd) + p.eps);
    var mg = 0.0;
    var mgx = 0.0;
    for (var c = 0u; c < hd; c = c + 1u) {
        let xh = (x[base + c] - mean) * inv;
        let g = dy[base + c] * gamma[c];
        mg = mg + g;
        mgx = mgx + g * xh;
    }
    mg = mg / f32(hd);
    mgx = mgx / f32(hd);
    for (var c = 0u; c < hd; c = c + 1u) {
        let xh = (x[base + c] - mean) * inv;
        let g = dy[base + c] * gamma[c];
        dx[base + c] = (g - mg - xh * mgx) * inv;
    }
}

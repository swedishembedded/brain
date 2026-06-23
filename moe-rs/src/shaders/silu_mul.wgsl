// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// SwiGLU activation core:  out[i] = SiLU(a[i]) * b[i],  SiLU(x) = x * sigmoid(x).
// Elementwise over seq_len * d_ff.

struct Params {
    total: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       a:   array<f32>;
@group(0) @binding(2) var<storage, read>       b:   array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.total) { return; }
    let x = a[idx];
    let silu = x / (1.0 + exp(-x));
    out[idx] = silu * b[idx];
}

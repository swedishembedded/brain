// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Strided per-head LayerNorm (QK-norm): normalize each length-`head_dim` head
// vector inside the q or k region of a fused [rows, row_stride] buffer, in
// place, with affine gamma/beta[head_dim]. One invocation per (row, head).
// WorldMirror applies this to q and k BEFORE RoPE.

struct Params {
    rows: u32,
    heads: u32,
    head_dim: u32,
    row_stride: u32, // e.g. 3*d_model for fused qkv
    off: u32,        // region offset: 0 = q, d_model = k
    eps: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> buf:   array<f32>;
@group(0) @binding(2) var<storage, read>       gamma: array<f32>;
@group(0) @binding(3) var<storage, read>       beta:  array<f32>;

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
    for (var c = 0u; c < hd; c = c + 1u) { mean = mean + buf[base + c]; }
    mean = mean / f32(hd);
    var va = 0.0;
    for (var c = 0u; c < hd; c = c + 1u) {
        let d = buf[base + c] - mean;
        va = va + d * d;
    }
    let inv = inverseSqrt(va / f32(hd) + p.eps);
    for (var c = 0u; c < hd; c = c + 1u) {
        buf[base + c] = (buf[base + c] - mean) * inv * gamma[c] + beta[c];
    }
}

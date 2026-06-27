// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Batched RoPE (forward), HF/Qwen "half-split" (GPT-NeoX) convention, with a
// configurable base theta. This differs from `rope_train.wgsl` (which rotates
// adjacent interleaved pairs `2j,2j+1`): here the rotated pair is the channel
// `m` and its partner `m + head_dim/2`, matching HF's `rotate_half` so imported
// Qwen weights behave identically. The base theta is a uniform (Qwen = 1e6).
//
// For position pos, angle θ_m = pos * base^(-2m/head_dim), m in [0, head_dim/2):
//   out[m]            = x[m]*cosθ - x[m+half]*sinθ
//   out[m+half]       = x[m+half]*cosθ + x[m]*sinθ
//
// Applied in place to a contiguous q or k buffer (one head-group per row),
// selected by row_stride/base_off. One invocation per (row, head, m).

struct Params {
    n_rows: u32,     // B*T
    n_heads: u32,    // heads in THIS buffer (n_heads for q, n_kv_heads for k)
    head_dim: u32,
    row_stride: u32, // per-row width of the buffer (n_heads*head_dim)
    base_off: u32,   // 0 (separate q/k buffers)
    tcols: u32,      // T
    rope_base: f32,  // rotary base theta (e.g. 1e6)
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> buf: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let half = p.head_dim / 2u;
    let total = p.n_rows * p.n_heads * half;
    let idx = gidx;
    if (idx >= total) { return; }

    let m = idx % half;
    let tmp = idx / half;
    let h = tmp % p.n_heads;
    let row = tmp / p.n_heads;
    let pos = row % p.tcols;

    let hbase = row * p.row_stride + p.base_off + h * p.head_dim;
    let angle = f32(pos) * pow(p.rope_base, -f32(2u * m) / f32(p.head_dim));
    let c = cos(angle);
    let s = sin(angle);
    let x0 = buf[hbase + m];
    let x1 = buf[hbase + m + half];
    buf[hbase + m]        = x0 * c - x1 * s;
    buf[hbase + m + half] = x1 * c + x0 * s;
}

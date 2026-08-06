// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  RoPE (forward) at an EXPLICIT absolute position — the decode-step twin of rope_base
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// RoPE (forward) at an EXPLICIT absolute position — the decode-step twin of
// rope_base. Identical math (HF/Qwen "half-split" GPT-NeoX, configurable base
// theta), but the rotary position is `pos_base + row` instead of `row % tcols`,
// so a single new token decoded at absolute position `pos_base` (n_rows = 1)
// rotates correctly against a KV cache that already holds positions 0..pos_base.
//
// For position pos, angle θ_m = pos * base^(-2m/head_dim), m in [0, head_dim/2):
//   out[m]      = x[m]*cosθ - x[m+half]*sinθ
//   out[m+half] = x[m+half]*cosθ + x[m]*sinθ
//
// Applied in place to a contiguous q or k buffer (one head-group per row).
// One invocation per (row, head, m). Barrier-free → JITs to CPU for free.

struct Params {
    n_rows: u32,     // rows in this call (1 for a decode step)
    n_heads: u32,    // heads in THIS buffer (n_heads for q, n_kv_heads for k)
    head_dim: u32,
    row_stride: u32, // per-row width (n_heads*head_dim)
    base_off: u32,   // 0 for separate q/k buffers
    pos_base: u32,   // absolute position of row 0
    rope_base: f32,  // rotary base theta (e.g. 1e6)
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> buf: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let half = p.head_dim / 2u;
    let total = p.n_rows * p.n_heads * half;
    let idx = gidx;
    if (idx >= total) { return; }

    let m = idx % half;
    let tmp = idx / half;
    let h = tmp % p.n_heads;
    let row = tmp / p.n_heads;
    let pos = p.pos_base + row;

    let hbase = row * p.row_stride + p.base_off + h * p.head_dim;
    let angle = f32(pos) * pow(p.rope_base, -f32(2u * m) / f32(p.head_dim));
    let c = cos(angle);
    let s = sin(angle);
    let x0 = buf[hbase + m];
    let x1 = buf[hbase + m + half];
    buf[hbase + m]        = x0 * c - x1 * s;
    buf[hbase + m + half] = x1 * c + x0 * s;
}

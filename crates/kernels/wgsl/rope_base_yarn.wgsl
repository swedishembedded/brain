// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Batched RoPE (forward), half-split (HF/Qwen) convention, reading a per-channel YaRN inv_freq table instead of an analytic base theta - the context-extension twin of rope_base
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Identical indexing and rotation to `rope_base.wgsl` (same `row % tcols`
// position scheme, same q/k row-stride/base-offset selection); the
// per-channel angular frequency comes from the `inv_freq` table rather than
// `pow(rope_base, -2m/head_dim)`, and both cos and sin are scaled by
// `attention_factor` (YaRN's mscale) before the rotation - the same table
// contract `rope_paged_yarn.wgsl` already established for the paged serving
// path.
//
// `inv_freq` is head_dim/2 long and position-independent: every row and head
// shares entry `m`. Build it with `model::yarn::scaled_inv_freq`, which
// returns the plain unscaled schedule (and `attention_factor = 1.0`) when
// scaling is disabled, so binding a table from a `factor <= 1.0` config
// reproduces `rope_base.wgsl` exactly - the same identity guarantee
// `YarnConfig`'s own doc gives.
//
// One invocation per (row, head, m<half).

struct Params {
    n_rows: u32,     // B*T
    n_heads: u32,    // heads in THIS buffer (n_heads for q, n_kv_heads for k)
    head_dim: u32,
    row_stride: u32, // per-row width of the buffer (n_heads*head_dim)
    base_off: u32,   // 0 (separate q/k buffers)
    tcols: u32,      // T
    attention_factor: f32, // YaRN mscale, applied to cos and sin
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> buf: array<f32>;
@group(0) @binding(2) var<storage, read>       inv_freq: array<f32>;

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
    let angle = f32(pos) * inv_freq[m];
    let c = cos(angle) * p.attention_factor;
    let s = sin(angle) * p.attention_factor;
    let x0 = buf[hbase + m];
    let x1 = buf[hbase + m + half];
    buf[hbase + m]        = x0 * c - x1 * s;
    buf[hbase + m + half] = x1 * c + x0 * s;
}

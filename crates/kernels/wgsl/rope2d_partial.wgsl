// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Table-driven interleaved M-RoPE, PARTIAL: rotate only the first `2*half` channels of each `head_dim`-wide head, in place on the q or k region of a fused [rows, row_stride] buffer
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// `rope2d.wgsl`'s table-driven rotation, but for a head whose rotated block is
// SHORTER than the head itself (Qwen3.5's `partial_rotary_factor`: only
// `rot_dim = head_dim * partial_rotary_factor` of each head is rotated, the
// remaining `head_dim - rot_dim` channels pass through untouched). The two
// differ only in how the per-head base offset is computed: `rope2d` derives it
// from `2*half` (assumes the whole head is rotated, so `half = head_dim/2`);
// here `half = rot_dim/2` is strictly smaller than `head_dim/2`, so the head
// stride must be passed separately as `head_dim`. The host precomputes
// per-token-position cos/sin tables `[tmod, half]` exactly as for `rope2d`
// (`qwenvl::mrope::mrope_tables` called with `head_dim = rot_dim`, since the
// table only ever spans the rotated sub-space). `sign` = 1 forward, -1 the
// exact inverse rotation (backward).
//   y1 = x1*cos - sign*x2*sin ;  y2 = x2*cos + sign*x1*sin

struct Params {
    rows: u32,
    heads: u32,
    half: u32,       // rot_dim / 2 (table width; rot_dim <= head_dim)
    row_stride: u32, // e.g. n_heads*head_dim for a plain q/k buffer
    off: u32,        // region offset: 0 = q, hq = k (or wherever the region starts)
    tmod: u32,       // table rows; token row -> table row = row % tmod
    sign: f32,
    head_dim: u32,   // full per-head stride; rot_dim = 2*half <= head_dim
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> buf: array<f32>;
@group(0) @binding(2) var<storage, read>       cos_t: array<f32>; // [tmod, half]
@group(0) @binding(3) var<storage, read>       sin_t: array<f32>; // [tmod, half]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.rows * p.heads * p.half;
    if (idx >= total) { return; }
    let d = idx % p.half;
    let r1 = idx / p.half;
    let h = r1 % p.heads;
    let row = r1 / p.heads;

    let t = (row % p.tmod) * p.half + d;
    let c = cos_t[t];
    let s = sin_t[t] * p.sign;
    // Full head_dim stride between heads (NOT 2*half) — this is the one line
    // that differs from rope2d.wgsl, and exactly why a bare constant bump
    // there would silently corrupt every head after the first.
    let base = row * p.row_stride + p.off + h * p.head_dim;
    let x1 = buf[base + d];
    let x2 = buf[base + d + p.half];
    buf[base + d] = x1 * c - x2 * s;
    buf[base + d + p.half] = x2 * c + x1 * s;
    // Channels [2*half, head_dim) of this head are never addressed above and
    // so pass through unrotated, matching rope_partial.wgsl's semantics.
}

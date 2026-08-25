// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Moondream partial RoPE at an explicit absolute position (decode step)
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The decode-step twin of rope_partial.wgsl, standing in exactly the relation to
// it that rope_at.wgsl stands to rope_base.wgsl: identical math, but the rotary
// position is `pos_base + row` instead of `row % tcols`.
//
// That difference is the whole reason this file exists. rope_partial derives the
// position from the ROW INDEX, which is right for a batched forward (row r is
// position r) and cannot express a single new token at position 137: with
// n_rows = 1 the only row is row 0, so `row % tcols` is 0 whatever `tcols` says.
// A decode step against a KV cache holding positions 0..pos needs the new row
// rotated at `pos`, not at 0 - and rotating it at 0 does not fail, it silently
// attends with the wrong angles.
//
// rope_at cannot be reused here: it rotates the FULL head_dim, while Moondream
// rotates only the first `rot_dim` of each head (32 of 64) and passes the rest
// through. Expressing that with rope_at would need one dispatch per head.
//
// Half-split within the rotated block: pair (m, m + rot_dim/2) for m in
// [0, rot_dim/2), angle θ_m = pos·base^(-2m/rot_dim). For Moondream rot_dim=32 of
// head_dim=64, base=1.5e6. In place on a q or k buffer (head stride = head_dim).
// One invocation per (row, head, m in [0, rot_dim/2)).

struct Params {
    n_rows: u32,     // rows in this call (1 for a decode step)
    n_heads: u32,
    head_dim: u32,
    row_stride: u32, // per-row width (3*d for a fused qkv buffer)
    base_off: u32,   // 0 for q, d for k within a fused qkv row
    pos_base: u32,   // absolute position of row 0
    rope_base: f32,
    rot_dim: u32,    // rotated channels per head; the rest pass through
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> buf: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let half = p.rot_dim / 2u;
    let total = p.n_rows * p.n_heads * half;
    if (gidx >= total) { return; }
    let m = gidx % half;
    let tmp = gidx / half;
    let h = tmp % p.n_heads;
    let row = tmp / p.n_heads;
    let pos = p.pos_base + row;

    let hbase = row * p.row_stride + p.base_off + h * p.head_dim;
    let angle = f32(pos) * pow(p.rope_base, -f32(2u * m) / f32(p.rot_dim));
    let c = cos(angle);
    let s = sin(angle);
    let x0 = buf[hbase + m];
    let x1 = buf[hbase + m + half];
    buf[hbase + m]        = x0 * c - x1 * s;
    buf[hbase + m + half] = x1 * c + x0 * s;
}

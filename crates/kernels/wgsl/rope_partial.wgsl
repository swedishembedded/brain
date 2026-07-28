// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Moondream partial RoPE (forward): rotate only the first `rot_dim` channels of
// each `head_dim`-wide head; the remaining `head_dim - rot_dim` pass through.
// Half-split within the rotated block: pair (m, m + rot_dim/2) for m in
// [0, rot_dim/2), angle θ_m = pos·base^(-2m/rot_dim). For Moondream rot_dim=32 of
// head_dim=64, base=1.5e6. In place on a q or k buffer (head stride = head_dim).
// One invocation per (row, head, m in [0, rot_dim/2)).

struct Params {
    n_rows: u32,
    n_heads: u32,
    head_dim: u32,
    row_stride: u32,
    base_off: u32,
    tcols: u32,
    rope_base: f32,
    rot_dim: u32,
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
    let pos = row % p.tcols;

    let hbase = row * p.row_stride + p.base_off + h * p.head_dim;
    let angle = f32(pos) * pow(p.rope_base, -f32(2u * m) / f32(p.rot_dim));
    let c = cos(angle);
    let s = sin(angle);
    let x0 = buf[hbase + m];
    let x1 = buf[hbase + m + half];
    buf[hbase + m]        = x0 * c - x1 * s;
    buf[hbase + m + half] = x1 * c + x0 * s;
}

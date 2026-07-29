// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// GQA head expansion into a fused attention buffer (LFM2.5 bidirectional path):
// replicate each of the src's `heads_out/group` kv heads `group` times,
// writing head ho from src head ho/group — the layout `repeat_kv` produces —
// so MHA attention kernels (n_heads == heads_out) run over GQA projections.
// `group == 1` is a plain strided copy (used to place q into the fused buffer).
//   dst[row*dst_stride + dst_off + ho*hd + d] = src[row*src_stride + (ho/group)*hd + d]
// One invocation per (row, ho, d); total = rows * heads_out * hd.
// Backward: kv_expand_bwd (group-sum). Pure copy — adjoint pairs are exact.

struct Params {
    rows: u32,
    heads_out: u32,
    group: u32,
    hd: u32,
    src_stride: u32,   // heads_out/group * hd
    dst_stride: u32,   // fused row width (3*d_model)
    dst_off: u32,      // region offset within the fused row
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.rows * p.heads_out * p.hd;
    if (idx >= total) { return; }

    let d = idx % p.hd;
    let r1 = idx / p.hd;
    let ho = r1 % p.heads_out;
    let row = r1 / p.heads_out;

    let hs = ho / p.group;
    dst[row * p.dst_stride + p.dst_off + ho * p.hd + d] =
        src[row * p.src_stride + hs * p.hd + d];
}

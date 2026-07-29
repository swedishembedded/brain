// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Backward of kv_expand — the adjoint of head replication is a group-sum:
//   d_src[row*src_stride + hs*hd + d] =
//       sum_{g<group} d_dst[row*dst_stride + dst_off + (hs*group+g)*hd + d]
// `group == 1` degenerates to the strided copy-out (q region extraction).
// One invocation per (row, hs, d); total = rows * (heads_out/group) * hd.
// Overwrites d_src (no accumulation; callers own any residual summing).

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
@group(0) @binding(1) var<storage, read>       d_dst: array<f32>;
@group(0) @binding(2) var<storage, read_write> d_src: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let heads_src = p.heads_out / p.group;
    let total = p.rows * heads_src * p.hd;
    if (idx >= total) { return; }

    let d = idx % p.hd;
    let r1 = idx / p.hd;
    let hs = r1 % heads_src;
    let row = r1 / heads_src;

    var acc = 0.0;
    for (var g: u32 = 0u; g < p.group; g = g + 1u) {
        let ho = hs * p.group + g;
        acc = acc + d_dst[row * p.dst_stride + p.dst_off + ho * p.hd + d];
    }
    d_src[row * p.src_stride + hs * p.hd + d] = acc;
}

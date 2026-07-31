// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// As head_pack but TRANSPOSED per head: out[ho*head_stride + d*rows + i] = src[...].
// The apply GEMM `ctx = probs @ V` runs as A·Bᵀ with B = Vᵀ[hd, rows], so V
// packs transposed. Same GQA head replication and scale semantics.
// One invocation per (ho, d, i); total = heads_out * rows * hd.

struct Params {
    rows: u32,
    heads_out: u32,
    group: u32,
    hd: u32,
    src_stride: u32,
    src_off: u32,
    scale: f32,
    head_stride: u32, // >= rows*hd, caller-padded so ho*head_stride lands storage-buffer-offset-aligned
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       src: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.heads_out * p.rows * p.hd;
    if (idx >= total) { return; }
    let i = idx % p.rows;
    let r1 = idx / p.rows;
    let d = r1 % p.hd;
    let ho = r1 / p.hd;
    out[ho * p.head_stride + d * p.rows + i] = src[i * p.src_stride + p.src_off + (ho / p.group) * p.hd + d] * p.scale;
}

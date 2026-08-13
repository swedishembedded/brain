// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Pack one attention operand head-major-contiguous for GEMM attention
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Pack one attention operand head-major-contiguous for GEMM attention:
//   out[ho*head_stride + i*hd + d] = src[i*src_stride + src_off + (ho/group)*hd + d] * scale
// Turns the row-major [rows, heads*hd] projection layout into per-head [rows, hd]
// blocks a register-tiled GEMM can consume, folding in the GQA head replication
// (group > 1 reads the narrow kv projection — no expanded buffer) and an
// optional scalar (1/sqrt(hd) on q folds the attention scale into the pack).
// One invocation per (ho, i, d); total = heads_out * rows * hd.

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
    let d = idx % p.hd;
    let r1 = idx / p.hd;
    let i = r1 % p.rows;
    let ho = r1 / p.rows;
    out[ho * p.head_stride + i * p.hd + d] = src[i * p.src_stride + p.src_off + (ho / p.group) * p.hd + d] * p.scale;
}

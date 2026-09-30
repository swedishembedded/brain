// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Copy a column range between row-major matrices of different width
// @how   one thread per copied element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Copy `cnt` columns of each of `rows` rows from a source matrix with row
// stride `src_stride` (starting at column `src_off`) into a destination with
// row stride `dst_stride` (starting at column `dst_off`):
//   dst[r * dst_stride + dst_off + j] = src[r * src_stride + src_off + j]
// Moves one vocab tile of a large matrix (the LM head's logits or their
// gradient) to or from a contiguous per-tile scratch, so the tile can go
// through the register-tiled GEMM kernels, which address dense operands.

struct Params {
    rows: u32,
    cnt: u32,
    src_stride: u32,
    src_off: u32,
    dst_stride: u32,
    dst_off: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.rows * p.cnt) { return; }
    let row = idx / p.cnt;
    let j = idx % p.cnt;
    dst[row * p.dst_stride + p.dst_off + j] = src[row * p.src_stride + p.src_off + j];
}

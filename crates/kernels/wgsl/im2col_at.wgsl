// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  im2col over a RANGE of output positions - `im2col.wgsl` with a `[pos0, pos0+cnt)` window, so a conv can be lowered to a GEMM in spatial chunks
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// im2col over a RANGE of output positions — `im2col.wgsl` with a `[pos0,
// pos0+cnt)` window, so a conv can be lowered to a GEMM in spatial chunks.
//
//   x   : [N=1, Cin, H, W]  (NCHW)
//   col : [cnt, Cin*K*K]    row-major — col[i, (ci*K + kh)*K + kw] = the input
//         pixel feeding output position `pos0 + i` for tap (ci,kh,kw), 0
//         outside padding.
//
// Why the window exists: the un-windowed operand for a 512x512 conv with
// Cin=256, K=3 is `[262144, 2304]` f32 = **2.4 GB**, over the P40's 2047 MiB
// `max_storage_buffer_binding_size`, so the whole-image im2col is not even
// bindable. Chunking the *rows* of the GEMM keeps one bounded scratch buffer
// and leaves the arithmetic identical: with `y[HW, Cout] = col . Wᵀ`, a
// position chunk is a contiguous row range of both `col` and `y`, so both
// bindings are plain sub-ranges.
//
// Threads are indexed by the col element (write-coalesced, gathering the input
// in 12-byte runs). A workgroup-staged 64x64 (position x tap) tile - the same
// shape that made `nlc_bias_nchw` several times faster - was tried here and
// measured **SLOWER, not faster**: unlike the transpose, this kernel's
// uncoalesced side is barely amplified (3 consecutive taps land in one
// sector), so the
// 16.6 KB of workgroup memory bought less than the occupancy it cost (5 blocks
// per SM instead of the shared-memory-free maximum). Kept element-indexed.
//
// One invocation per (i, cinkk) element of the window.

struct Params {
    cin: u32, h: u32, w: u32, k: u32,
    stride: u32, pad: u32, ho: u32, wo: u32,
    cinkk: u32,      // Cin*K*K (col row stride)
    pos0: u32,       // first output spatial index of this window
    cnt: u32,        // number of positions in this window
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read_write> col: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.cnt * p.cinkk;
    if (idx >= total) { return; }

    let cinkk = idx % p.cinkk;
    let pos = p.pos0 + idx / p.cinkk;   // absolute output spatial index
    let ho = pos / p.wo;
    let wo = pos % p.wo;
    let kk = p.k * p.k;
    let ci = cinkk / kk;
    let r = cinkk % kk;
    let kh = r / p.k;
    let kw = r % p.k;

    // signed source coords (i32 to catch the negative-padding region)
    let ih = i32(ho * p.stride + kh) - i32(p.pad);
    let iw = i32(wo * p.stride + kw) - i32(p.pad);
    var v = 0.0;
    if (ih >= 0 && iw >= 0 && ih < i32(p.h) && iw < i32(p.w)) {
        v = x[(ci * p.h + u32(ih)) * p.w + u32(iw)];
    }
    col[idx] = v;
}

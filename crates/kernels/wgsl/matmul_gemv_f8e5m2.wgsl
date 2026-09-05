// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Skinny-M matmul with a portable FP8 E5M2 weight (out = x @ dequant_e5m2(w)ᵀ), BLOCKWISE 128x128 scale, one WORKGROUP per output COLUMN
// @how   64-thread workgroup tile, 1 barrier, serial inner reduction
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// M8.6's other portable FP8 storage tier - `matmul_gemv_f8e4m3.wgsl`'s exact
// structure (see that kernel's own header for the full blockwise-scale
// derivation and the W4A8-vs-storage-tier distinction) with the OTHER fixed
// 8-bit codebook: E5M2 (OCP FP8 - has real infinities, unlike E4M3FN). The
// decode expression is `kernels::template::f8e5m2_decode_expr`'s exact
// output, pasted verbatim - see that function's own doc for the bit
// derivation (structurally `f16_decode_expr` narrowed to a 2-bit mantissa,
// same 5-bit exponent width/bias) and its module's exhaustive 256-byte-
// pattern test.
//
//   x     : [M, K]                  f32 - plain, unquantized
//   wq    : [N, ceil(K/4)]          u32 - 4 packed E5M2 bytes per word, byte `b` of word `w` covers element `w*4+b`. REQUIRES k % 4 == 0.
//   scale : [ceil(N/128)*cb]        f32 - row-major `(row_block, col_block)` grid, `cb = ceil(K/128)` (passed in `p.cb`)
//   out   : [M, N]                  f32
//   params: m, k, n, cb. REQUIRES m <= 32.

struct Params {
    m: u32,
    k: u32,
    n: u32,
    cb: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:     array<f32>;  // [M, K]
@group(0) @binding(2) var<storage, read>       wq:    array<u32>;  // [N, ceil(K/4)]
@group(0) @binding(3) var<storage, read>       scale: array<f32>;  // [ceil(N/128)*cb]
@group(0) @binding(4) var<storage, read_write> out:   array<f32>;  // [M, N]

var<workgroup> partial: array<f32, 2048>; // up to 32 rows x 64 threads

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let col = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (col >= p.n) { return; }
    for (var m = 0u; m < p.m; m = m + 1u) {
        partial[m * 64u + t] = 0.0;
    }
    let kw = p.k / 4u; // words per row (4 E5M2 bytes/word)
    let wbase = col * kw;
    let rb = col / 128u;
    for (var k = t; k < p.k; k = k + 64u) {
        let word = wq[wbase + k / 4u];
        let byte = (word >> (8u * (k & 3u))) & 0xFFu;
        let wv = bitcast<f32>(bitcast<u32>(select(select((bitcast<f32>(0x38800000u | (((byte & 0x3u)) << 21u)) - bitcast<f32>(0x38800000u)), (bitcast<f32>(((byte & 0x7Fu)) << 21u) * bitcast<f32>(0x77800000u)), (((byte >> 2u) & 0x1Fu)) != 0u), bitcast<f32>(0x7F800000u | (((byte & 0x3u)) << 21u)), (((byte >> 2u) & 0x1Fu)) == 31u)) | ((byte & 0x80u) << 24u));
        let bc = k / 128u;
        let s = scale[rb * p.cb + bc];
        let wvs = wv * s;
        for (var m = 0u; m < p.m; m = m + 1u) {
            partial[m * 64u + t] = partial[m * 64u + t] + x[m * p.k + k] * wvs;
        }
    }
    workgroupBarrier();
    if (t < p.m) {
        var s = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) {
            s = s + partial[t * 64u + i];
        }
        out[t * p.n + col] = s;
    }
}

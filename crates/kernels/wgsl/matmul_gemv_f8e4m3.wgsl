// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Skinny-M matmul with a portable FP8 E4M3 weight (out = x @ dequant_e4m3(w)ᵀ), BLOCKWISE 128x128 scale, one WORKGROUP per output COLUMN
// @how   64-thread workgroup tile, 1 barrier, serial inner reduction
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant none
// @dtype f32
//
// M8.6: a portable (decode-to-f32)
// FP8 STORAGE tier, structurally `matmul_gemv.wgsl` (the plain f32
// decode-regime GEMV that kernel's own header documents in full) with the
// weight read as packed E4M3 bytes instead of f32, decoded inline. NOT a
// templated `#w=...` variant (`kernels::template::dtype_variant`'s
// declaration/load rewrite is hardcoded to 2-per-`u32` packing with no
// second binding - neither fits FP8's 4-per-`u32` packing AND its extra
// BLOCKWISE scale binding), so this is hand-written; the decode expression
// itself is `kernels::template::f8e4m3_decode_expr`'s exact output, pasted
// verbatim (see that function's own doc for the full bit derivation and its
// module's exhaustive 256-byte-pattern test) - a
// `f8e4m3_decode_expr_matches_this_kernels_pasted_copy` test in that same
// module pins the two can never silently drift.
//
// Unlike EVERY other quantized/storage tier in this tree, the scale is
// BLOCKWISE, not per-tensor/per-row/per-group-of-K: one `f32` per `128x128`
// block of the LOGICAL `[N, K]` weight (`model::fp8::scale_shape`'s shape -
// the real DeepSeek-V3/Qwen3.5-FP8 checkpoint convention this format
// mirrors). Since one workgroup owns exactly one weight ROW (`col`, fixed
// for the whole dispatch), the ROW block index is a per-workgroup constant
// (`rb = col / 128u`); the COLUMN block index advances every 128 elements of
// the reduction axis as the k-loop runs (`bc = k / 128u`). `matmul_kq_dyn.wgsl`
// is the nearest existing structural sibling for "more than one scale array
// indexed nontrivially", though that kernel's own two scales are both
// GROUP/super-block (1-D) indexed, not this kernel's genuine 2-D
// `(row_block, col_block)` grid - there was no existing 2-D-indexed scale to
// copy, so this indexing is new, not adapted.
//
// Activations stay plain, UNQUANTIZED f32 - this is a decode-to-f32 STORAGE
// tier like `BF16`/`F16`, not W4A8 like `Q4`/`NF4`/`F4E2M1`, so there is no
// int8 activation quant step to reuse.
//
//   x     : [M, K]                  f32 - plain, unquantized
//   wq    : [N, ceil(K/4)]          u32 - 4 packed E4M3 bytes per word, byte `b` of word `w` covers element `w*4+b`. REQUIRES k % 4 == 0.
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
    let kw = p.k / 4u; // words per row (4 E4M3 bytes/word)
    let wbase = col * kw;
    let rb = col / 128u;
    for (var k = t; k < p.k; k = k + 64u) {
        let word = wq[wbase + k / 4u];
        let byte = (word >> (8u * (k & 3u))) & 0xFFu;
        let wv = bitcast<f32>(bitcast<u32>(select(select((bitcast<f32>(0x3C800000u | (((byte & 0x7Fu)) << 20u)) - bitcast<f32>(0x3C800000u)), (bitcast<f32>(((byte & 0x7Fu)) << 20u) * bitcast<f32>(0x7B800000u)), ((byte & 0x7Fu)) >= 8u), bitcast<f32>(0x7FC00000u), ((byte & 0x7Fu)) == 0x7Fu)) | ((byte & 0x80u) << 24u));
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

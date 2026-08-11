// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Skinny-M W4A8 matmul (out = dequant(x_q8 @ w_q4ᵀ)), one WORKGROUP per output COLUMN -- the decode-regime q4 GEMM
// @how   64-thread workgroup tile, 1 barrier, serial inner reduction
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant q4
//
// Skinny-M W4A8 matmul (out = dequant(x_q8 @ w_q4^T)), one WORKGROUP per
// output COLUMN -- the decode-regime q4 GEMM, mirroring
// `matmul_i8_gemv.wgsl`'s shape exactly (same Params order, same 64-thread
// workgroup-per-column tile, same `partial[m*64+t]` shared-memory layout, same
// single top-level barrier) with int4 weights instead of int8:
//
//   x_q : [M, K/4] u32  -- 4 int8 activations packed along K per u32
//   w_q : [N, K/8] u32  -- 8 int4 weights    packed along K per u32
//   sx  : [M] per-token activation scale     sw : [N] per-channel weight scale
//   out : [M, N] f32    -- out[m,n] = acc_i32 * sx[m] * sw[n]
//   params: m, k (LOGICAL K, a multiple of 8). REQUIRES m <= 32.
//
// Why this exists: the same reason `matmul_i8_gemv` exists next to
// `matmul_i8_dyn` -- at decode M is 1..32 and a wide tile is mostly idle. This
// gives q4 the identical decode-regime shape int8 already has: 64 threads
// split the packed K axis, each reads its slice of W row `col` ONCE and
// applies it to all M rows, one barrier, threads 0..m fold the partials.
//
// NOT register-tiled (`.agents/rules/porting.md` §10: correct, then freeze) --
// this is the naive-but-cooperative tier, same complexity class as
// `matmul_i8_gemv`, not `matmul_i8_dyn`'s 128x128 interleaved shape.
//
// Nibble/byte sign extension uses `shl` + arithmetic `shr`, not the WGSL
// builtin `extractBits` -- see `matmul_q4_dyn.wgsl`'s header for why
// (`extractBits` / `MathFunction::ExtractBits` has no lowering in this
// repo's CPU Cranelift JIT). Inlined into `main`, not a helper `fn` -- same
// reason as `matmul_q4_dyn.wgsl`'s header: the CPU JIT has no lowering for
// calling a user-defined WGSL function, and no other kernel in this tree
// defines one.

struct Params {
    m: u32,
    k: u32,
    n: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       xq:  array<u32>;  // [M, k/4]
@group(0) @binding(2) var<storage, read>       wq:  array<u32>;  // [N, k/8]
@group(0) @binding(3) var<storage, read>       sx:  array<f32>;  // [M]
@group(0) @binding(4) var<storage, read>       sw:  array<f32>;  // [N]
@group(0) @binding(5) var<storage, read_write> out: array<f32>;  // [M, N]

// i32 accumulators in workgroup memory (indexed [m*64 + t]) -- same layout as
// matmul_i8_gemv, same CPU-JIT-compatible single-barrier shape.
var<workgroup> partial: array<i32, 2048>; // up to 32 rows x 64 threads

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let col = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (col >= p.n) { return; }
    for (var m = 0u; m < p.m; m = m + 1u) {
        partial[m * 64u + t] = 0;
    }
    let kgx = p.k / 4u; // x words per row (int8 packing)
    let kgw = p.k / 8u; // w words per row (int4 packing)
    let wbase = col * kgw;
    for (var g = t; g < kgw; g = g + 64u) {
        let wv = wq[wbase + g];
        for (var m = 0u; m < p.m; m = m + 1u) {
            let xbase = m * kgx + 2u * g;
            let xw0 = xq[xbase];
            let xw1 = xq[xbase + 1u];
            var local = 0i;
            for (var b: u32 = 0u; b < 8u; b = b + 1u) {
                let wn = bitcast<i32>(wv << (28u - 4u * b)) >> 28u;
                var xb: i32;
                if (b < 4u) {
                    xb = bitcast<i32>(xw0 << (24u - 8u * b)) >> 24u;
                } else {
                    let bb = b - 4u;
                    xb = bitcast<i32>(xw1 << (24u - 8u * bb)) >> 24u;
                }
                local = local + wn * xb;
            }
            partial[m * 64u + t] = partial[m * 64u + t] + local;
        }
    }
    workgroupBarrier();
    // Threads 0..m each fold one row's 64 partials and dequantize.
    if (t < p.m) {
        var s = 0;
        for (var i = 0u; i < 64u; i = i + 1u) {
            s = s + partial[t * 64u + i];
        }
        out[t * p.n + col] = f32(s) * sx[t] * sw[col];
    }
}

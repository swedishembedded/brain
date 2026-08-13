// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Skinny-M INT8 matmul (out = dequant(x_q @ W_qᵀ)), one WORKGROUP per output COLUMN - the decode-regime int8 GEMM
// @how   DP4A packed int8, 64-thread workgroup tile, 1 barrier
// @opt   5
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant int8
// @dtype f32
//
// Skinny-M INT8 matmul (out = dequant(x_q @ W_qᵀ)), one WORKGROUP per output
// COLUMN — the decode-regime int8 GEMM.
//
//   x_q : [M, K/4] u32  — 4 int8 activations packed along K per u32
//   w_q : [N, K/4] u32  — 4 int8 weights    packed along K per u32
//   sx  : [M] per-token activation scale    sw : [N] per-channel weight scale
//   out : [M, N] f32    — out[m,n] = acc_i32 * sx[m] * sw[n]
//   params: m, kg (=K/4), n. REQUIRES m <= 32.
//
// Why this exists: the tiled `matmul_i8_dyn` owns the large-M regime (prefill),
// but at decode M is 1–32 and a 128x128 tile is mostly idle — measured 78 tok/s
// at c=1 against the fp32 GEMV's 127. This kernel gives int8 the same shape the
// fp32 decode path has (`matmul_gemv`): 64 threads split the packed K axis,
// each reads its slice of W row `col` ONCE and applies it to all M rows via
// dot4I8Packed, one barrier, threads 0..m fold the partials. W traffic drops
// M-fold, stays coalesced, and is 4x smaller than fp32 on top.
//
// Single top-level barrier + no atomics — the engine's portable reduction
// idiom. Dispatch: n * 64 invocations (one workgroup per column).

struct Params {
    m: u32,
    kg: u32,
    n: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       xq:  array<u32>;  // [M, kg]
@group(0) @binding(2) var<storage, read>       wq:  array<u32>;  // [N, kg]
@group(0) @binding(3) var<storage, read>       sx:  array<f32>;  // [M]
@group(0) @binding(4) var<storage, read>       sw:  array<f32>;  // [N]
@group(0) @binding(5) var<storage, read_write> out: array<f32>;  // [M, N]

// i32 accumulators in workgroup memory (indexed [m*64 + t]) — same layout as
// matmul_gemv, same CPU-JIT-compatible single-barrier shape.
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
    let wbase = col * p.kg;
    for (var g = t; g < p.kg; g = g + 64u) {
        let wv = wq[wbase + g];
        for (var m = 0u; m < p.m; m = m + 1u) {
            partial[m * 64u + t] = partial[m * 64u + t] + dot4I8Packed(xq[m * p.kg + g], wv);
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

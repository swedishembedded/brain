// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Skinny-M W4A8 matmul with the OCP MXFP4 E2M1 codebook (out = dequant_f4e2m1(x_q8 @ w_f4ᵀ)), one WORKGROUP per output COLUMN
// @how   64-thread workgroup tile, 1 barrier, serial inner reduction
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant q4
// @dtype f32
//
// M8.5: `matmul_q4_gemv_nf4.wgsl`'s
// exact structure with a DIFFERENT fixed codebook - the OCP Microscaling FP4
// format (E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit, magnitudes
// `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`, `model::lut4::F4E2M1_LUT`). See that
// kernel's own header for the full derivation of why the accumulation is a
// per-nibble f32 MAC (`local = sum_b LUT[code_b] * x_b`, scale pulled out
// once per word) rather than an i32 dot-then-scale - identical reasoning,
// this codebook is exactly as non-linear in the raw code as NF4's.
//
//   x_q : [M, K/4] u32  -- 4 int8 activations packed along K per u32
//   w_q : [N, K/8] u32  -- 8 F4E2M1 codes (UNSIGNED, 0..16) packed along K per u32
//   sx  : [M] per-token activation scale
//   sw  : [N, K/32] GROUP-WISE weight scale (`model::int8::GROUP`)
//   out : [M, N] f32    -- out[m,n] = sx[m] * sum_g F4E2M1_LUT[code] * sw[n,g] * x[m,...]
//   params: m, k (LOGICAL K, a multiple of 32). REQUIRES m <= 32.
//
// The codebook lookup is the same 4-level binary `select` tree shape
// `matmul_q4_gemv_nf4.wgsl` uses, on the SAME nibble bits, just the 16
// literal float values swapped for `F4E2M1_LUT`'s own (bit 3 is the sign,
// bits 0..2 pick one of the 8 magnitudes - see `model::lut4::F4E2M1_LUT`'s
// own doc comment for the derivation).

struct Params {
    m: u32,
    k: u32,
    n: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       xq:  array<u32>;  // [M, k/4]
@group(0) @binding(2) var<storage, read>       wq:  array<u32>;  // [N, k/8]
@group(0) @binding(3) var<storage, read>       sx:  array<f32>;  // [M]
@group(0) @binding(4) var<storage, read>       sw:  array<f32>;  // [N, k/32]
@group(0) @binding(5) var<storage, read_write> out: array<f32>;  // [M, N]

// Packed u32 words of w per weight-scale group: GROUP(32 f4e2m1 codes) / 8 per word.
const WPG4: u32 = 4u;

// f32 accumulators in workgroup memory (indexed [m*64 + t]) -- same layout as
// matmul_q4_gemv, same CPU-JIT-compatible single-barrier shape.
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
    let kgx = p.k / 4u; // x words per row (int8 packing)
    let kgw = p.k / 8u; // w words per row (f4e2m1 packing)
    let wbase = col * kgw;
    let swbase = col * (kgw / WPG4);
    for (var g = t; g < kgw; g = g + 64u) {
        let wv = wq[wbase + g];
        let s = sw[swbase + g / WPG4];
        for (var m = 0u; m < p.m; m = m + 1u) {
            let xbase = m * kgx + 2u * g;
            let xw0 = xq[xbase];
            let xw1 = xq[xbase + 1u];
            var local = 0.0;
            for (var b: u32 = 0u; b < 8u; b = b + 1u) {
                let code = (wv >> (4u * b)) & 0xFu;
                let wn = select(
                    select(
                        select(select(0.0, 0.5, (code & 1u) != 0u), select(1.0, 1.5, (code & 1u) != 0u), (code & 2u) != 0u),
                        select(select(2.0, 3.0, (code & 1u) != 0u), select(4.0, 6.0, (code & 1u) != 0u), (code & 2u) != 0u),
                        (code & 4u) != 0u),
                    select(
                        select(select(-0.0, -0.5, (code & 1u) != 0u), select(-1.0, -1.5, (code & 1u) != 0u), (code & 2u) != 0u),
                        select(select(-2.0, -3.0, (code & 1u) != 0u), select(-4.0, -6.0, (code & 1u) != 0u), (code & 2u) != 0u),
                        (code & 4u) != 0u),
                    (code & 8u) != 0u);
                var xb: i32;
                if (b < 4u) {
                    xb = bitcast<i32>(xw0 << (24u - 8u * b)) >> 24u;
                } else {
                    let bb = b - 4u;
                    xb = bitcast<i32>(xw1 << (24u - 8u * bb)) >> 24u;
                }
                local = local + wn * f32(xb);
            }
            partial[m * 64u + t] = partial[m * 64u + t] + local * s;
        }
    }
    workgroupBarrier();
    // Threads 0..m each fold one row's 64 partials and apply the per-token
    // activation scale (the weight side is already in).
    if (t < p.m) {
        var s = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) {
            s = s + partial[t * 64u + i];
        }
        out[t * p.n + col] = s * sx[t];
    }
}

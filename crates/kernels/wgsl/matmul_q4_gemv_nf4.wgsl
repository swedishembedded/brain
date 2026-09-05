// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Skinny-M W4A8 matmul with the bitsandbytes NF4 codebook (out = dequant_nf4(x_q8 @ w_nf4ᵀ)), one WORKGROUP per output COLUMN
// @how   64-thread workgroup tile, 1 barrier, serial inner reduction
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   yes
// @quant q4
// @dtype f32
//
// M8.5: `matmul_q4_gemv.wgsl` with
// its ONE change spelled out - the per-nibble weight value is a lookup into
// the fixed 16-entry NF4 codebook (`model::lut4::NF4_LUT`, reproduced below
// as a `select` tree, ascending order, code 0..16 unsigned) instead of the
// nibble's own sign-extended integer value. Everything else - packing,
// scale-group layout, workgroup shape, barrier count - is `matmul_q4_gemv`'s
// unchanged (see that kernel's own header for the full W4A8 layout/params
// doc, not repeated here):
//
//   x_q : [M, K/4] u32  -- 4 int8 activations packed along K per u32
//   w_q : [N, K/8] u32  -- 8 NF4 codes (UNSIGNED, 0..16) packed along K per u32
//   sx  : [M] per-token activation scale
//   sw  : [N, K/32] GROUP-WISE weight scale (`model::int8::GROUP`)
//   out : [M, N] f32    -- out[m,n] = sx[m] * sum_g NF4_LUT[code] * sw[n,g] * x[m,...]
//   params: m, k (LOGICAL K, a multiple of 32). REQUIRES m <= 32.
//
// **Why the accumulator is f32 MAC per nibble, not an i32 dot-then-scale
// like `matmul_q4_gemv`.** `Q4`'s uniform codebook is LINEAR in the raw
// signed nibble (`value == code`), so `sum_b(code_b * x_b) * scale` equals
// `sum_b(code_b * scale * x_b)` and the scale can be pulled out of the whole
// 8-wide reduction, letting the inner loop accumulate in exact i32 first.
// NF4's codebook is NOT linear in the raw code - `LUT[code]` has no closed
// form in `code` - so each nibble's dequantized value must be looked up
// BEFORE it is multiplied by the activation. The scale is still constant
// across all 8 nibbles of one word (`WPG4=4` words/group, same as
// `matmul_q4_gemv`), so it is still pulled out of the 8-wide reduction, just
// one level higher: `local = sum_b(LUT[code_b] * x_b)`, `partial += local *
// s`. This computes the exact same real number `Q4`'s formula would if `Q4`
// used this codebook - not an approximation of it.
//
// The codebook lookup is a 4-level binary `select` tree on the nibble's own
// bits (`code & 1u`/`2u`/`4u`/`8u`), NOT an `array<f32,16>` const - this
// repo's WGSL kernels avoid indexed const arrays entirely (untested on the
// CPU JIT; every existing lookup in this tree, `f16_decode_expr` included,
// is `select`-based), and a 4-level binary tree of scalar `select`s is exactly
// as portable as the `bitcast`/`select` machinery those kernels already use.
// Nibble/byte extraction otherwise identical to `matmul_q4_gemv.wgsl` - shl +
// arithmetic shr for the SIGNED int8 activation byte (unchanged from that
// kernel); the weight nibble itself is read UNSIGNED (`& 0xFu`, no sign
// extension - NF4_LUT's own values already carry the sign).

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

// Packed u32 words of w per weight-scale group: GROUP(32 nf4 codes) / 8 per word.
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
    let kgw = p.k / 8u; // w words per row (nf4 packing)
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
                        select(select(-1.0, -0.6961928009986877, (code & 1u) != 0u), select(-0.5250730514526367, -0.39491748809814453, (code & 1u) != 0u), (code & 2u) != 0u),
                        select(select(-0.28444138169288635, -0.18477343022823334, (code & 1u) != 0u), select(-0.09105003625154495, 0.0, (code & 1u) != 0u), (code & 2u) != 0u),
                        (code & 4u) != 0u),
                    select(
                        select(select(0.07958029955625534, 0.16093020141124725, (code & 1u) != 0u), select(0.24611230194568634, 0.33791524171829224, (code & 1u) != 0u), (code & 2u) != 0u),
                        select(select(0.44070982933044434, 0.5626170039176941, (code & 1u) != 0u), select(0.7229568362236023, 1.0, (code & 1u) != 0u), (code & 2u) != 0u),
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

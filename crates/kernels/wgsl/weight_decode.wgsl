// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training and inference over
// quantized checkpoints for its clients. If your team needs expertise in
// running tensor-core GEMMs over packed weights without ever expanding a whole
// tensor to full precision, you can procure our services by sending an email
// to info@swedishembedded.com.

// @what  Decode a window of rows of any packed weight tier (bf16, f16, int8, int4, NF4, FP8 E4M3/E5M2, K-quant) to a dense f32 tile
// @how   one thread per decoded element, format chosen by the FMT template constant
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The decode half of the quantized-execution contract of `model::ops`: a
// frozen quantized weight is multiplied (forward `Y = X W^T`, input gradient
// `dX = dY W`) by decoding a bounded tile of rows to f32 here and running the
// dense GEMM on the tile, so the scratch is `tile_rows * k * 4` bytes whatever
// the size of the tensor. One kernel for every tier: the stored layouts are the
// ones of `model::{int8,int4,lut4,fp8,kquant}` and `Weight::upload`, and the
// `FMT` constant (rewritten per variant by `kernels::template::weight_decode_variants`)
// selects the decode, so a new tier is one more arm here, not a new kernel
// family. Decoded values are exactly `host decode * scale` (no rounding to a
// narrower type), so the tile equals the host reference bit for bit.
//
//   FMT 0 bf16     w: flat [n*k] bf16, two per word
//   FMT 1 f16      w: flat [n*k] f16, two per word
//   FMT 2 int8     w: [n, k/4] signed bytes;           s: [n, k/32] f32
//   FMT 3 int4     w: [n, k/8] signed nibbles;         s: [n, k/32] f32
//   FMT 4 nf4      w: [n, k/8] codes;                  s: [n, k/32] f32
//   FMT 5 fp8 e4m3 w: [n, k/4] bytes;  s: [ceil(n/B), cb] f32, B = p.block, cb = ceil(k/B)
//   FMT 6 fp8 e5m2 same layout as 5
//   FMT 7 kq q4_k  w: [n, k/8] unsigned codes; wsm/wd: packed (sc, m) / (d, dmin)
//   FMT 8 kq q5_k  w: [n, k/4] unsigned codes; wsm/wd as FMT 7
//   FMT 9 kq q6_k  w: [n, k/4] signed bytes;           s: [n, k/16] f32
//
//   params: row0 (first row decoded), rows (rows in the tile), k, block, cb,
//           wsm words per row, wd words per row, padding.
//   out   : [rows, k] f32, row r = weight row row0 + r.

struct Params {
    row0: u32,
    rows: u32,
    k: u32,
    block: u32,
    cb: u32,
    sm_per_row: u32,
    wd_per_row: u32,
    pad: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       w:   array<u32>;
@group(0) @binding(2) var<storage, read>       s:   array<f32>;
@group(0) @binding(3) var<storage, read>       wsm: array<u32>;
@group(0) @binding(4) var<storage, read>       wd:  array<u32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

const FMT: u32 = 0u;

//@NF4_LUT@

fn f16_to_f32(h: u32) -> f32 {
    let sign = (h & 0x8000u) << 16u;
    let exp = (h >> 10u) & 0x1Fu;
    let subnormal = bitcast<f32>(0x38800000u | ((h & 0x3FFu) << 13u)) - bitcast<f32>(0x38800000u);
    let normal = bitcast<f32>((h & 0x7FFFu) << 13u) * bitcast<f32>(0x77800000u);
    let inf_or_nan = bitcast<f32>(0x7F800000u | ((h & 0x3FFu) << 13u));
    let mag = select(select(subnormal, normal, exp != 0u), inf_or_nan, exp == 31u);
    return bitcast<f32>(bitcast<u32>(mag) | sign);
}

//@FP8_FNS@

fn sext8(b: u32) -> f32 {
    return f32(bitcast<i32>(b << 24u) >> 24u);
}

fn sext4(b: u32) -> f32 {
    return f32(bitcast<i32>(b << 28u) >> 28u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.rows * p.k) { return; }
    let r = idx / p.k;
    let c = idx % p.k;
    let row = p.row0 + r;
    let wi = row * p.k + c;
    var v = 0.0;
    if (FMT == 0u) {
        v = bitcast<f32>(((w[wi >> 1u] >> ((wi & 1u) * 16u)) & 0xFFFFu) << 16u);
    } else if (FMT == 1u) {
        v = f16_to_f32((w[wi >> 1u] >> ((wi & 1u) * 16u)) & 0xFFFFu);
    } else if (FMT == 2u) {
        v = sext8((w[wi >> 2u] >> ((wi & 3u) * 8u)) & 0xFFu) * s[wi >> 5u];
    } else if (FMT == 3u) {
        v = sext4((w[wi >> 3u] >> ((wi & 7u) * 4u)) & 0xFu) * s[wi >> 5u];
    } else if (FMT == 4u) {
        v = nf4_lut((w[wi >> 3u] >> ((wi & 7u) * 4u)) & 0xFu) * s[wi >> 5u];
    } else if (FMT == 5u) {
        v = fp8_e4m3((w[wi >> 2u] >> ((wi & 3u) * 8u)) & 0xFFu) * s[(row / p.block) * p.cb + c / p.block];
    } else if (FMT == 6u) {
        v = fp8_e5m2((w[wi >> 2u] >> ((wi & 3u) * 8u)) & 0xFFu) * s[(row / p.block) * p.cb + c / p.block];
    } else if (FMT == 9u) {
        v = sext8((w[wi >> 2u] >> ((wi & 3u) * 8u)) & 0xFFu) * s[wi >> 4u];
    } else {
        // Affine K-quant (FMT 7: 4-bit codes, FMT 8: 8-bit slots): ds * code - dm per 32-element group.
        var code = 0.0;
        if (FMT == 7u) {
            code = f32((w[wi >> 3u] >> ((wi & 7u) * 4u)) & 0xFu);
        } else {
            code = f32((w[wi >> 2u] >> ((wi & 3u) * 8u)) & 0xFFu);
        }
        let g = c / 32u;
        let wword = wd[row * p.wd_per_row + g / 8u];
        let sword = wsm[row * p.sm_per_row + g / 2u];
        let shift = (g % 2u) * 16u;
        let ds = f16_to_f32(wword & 0xFFFFu) * f32((sword >> shift) & 0xFFu);
        let dm = f16_to_f32(wword >> 16u) * f32((sword >> (shift + 8u)) & 0xFFu);
        v = ds * code - dm;
    }
    out[idx] = v;
}

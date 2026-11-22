// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving a model with the low-precision activation numerics it was
// trained with for its clients. If your team needs expertise in NVFP4 and MXFP8 microscaled inference then you can
// procure our services by sending an email to info@swedishembedded.com.

// @what  Round-trip activation rows through a declared microscaled format (NVFP4 per-token four-over-six MAE, or MXFP8): out = dequant(quant(x))
// @how   64-thread workgroup per row, a lane per scale block, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant nvfp4
// @dtype f32
//
// What a release's runtime does to the activation operand of a product whose activations it declares quantized
// (`required_activation_quantization`), as values: the f32 a W4A4 / W8A8 product multiplies after decoding its
// activation codes. Feeding these rows to the f32-activation product of the same weights is the quantized product up to
// the accumulation order. Bit-identical to `model::hostmath::nvfp4_four_over_six_qdq` / `mxfp8_qdq`.
//
// scheme 0, NVFP4 with per-token amax and the four-over-six MAE block-scale rule (16-element blocks, E4M3 block scales):
//   amax = max |x| over the row; a row with amax < 2^-100 (all zero in practice) is all zero.
//   enc = 1536 / amax; dec = 1 / enc; outer = amax * (1/1536)      (1536 = 6 * 256: the E4M3 ceiling of a 6-based
//                                                                   scale is 256 so its 1.5x four-based twin fits 448)
//   per block, bmax = max |x|:  s6 = e4m3(min(bmax * (enc * (1/6)), 448)), s4 = e4m3(min(bmax * (enc * 0.25), 448))
//   candidate c: v = e2m1(x * (1 / (s_c * dec))) (nearest, ties to even, saturating at 6), r = (v * s_c) * outer
//   (a zero s_c reconstructs zero); err_c = sum |r - x| as a pairwise tree over the 16 lanes;
//   the four-based candidate is kept only when err4 < err6 strictly.
// scheme 1, MXFP8 (32-element blocks, E8M0 scale, E4M3 elements):
//   raw = bmax * (1/448); the scale is raw rounded UP to a power of two (2^ceil(log2 raw)), read off raw's bits;
//   a block whose raw is not a normal float (bmax < 448 * 2^-126) is all zero.
//   r = e4m3(clamp(x / scale, -448, 448)) * scale   (nearest, ties to even)
//
// Rows hold finite values. The gradient of this rounding is the straight-through identity: a backward passes dy as dx,
// so no backward kernel exists.
//
//   x : [rows, k] f32   out : [rows, k] f32   (k a multiple of the scheme's block)
//   params : u32 [rows, k, scheme]

struct Params {
    rows: u32,
    k: u32,
    scheme: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

var<workgroup> partial: array<f32, 64>;

const TINY: f32 = 7.888609e-31; // 2^-100
const MIN_NORMAL: f32 = 1.1754944e-38; // 2^-126

// The E4M3 magnitude code nearest `y` (`0 <= y <= 448`), ties to even.
fn e4m3_magnitude(y: f32) -> u32 {
    if (y < 0.015625) { return u32(round(y * 512.0)); }
    let bits = bitcast<u32>(y);
    let r = bits + 0x7FFFFu + ((bits >> 20u) & 1u);
    return min((r >> 20u) - 960u, 126u);
}

// The value of an E4M3 magnitude code.
fn e4m3_value(m: u32) -> f32 {
    if (m < 8u) { return f32(m) / 512.0; }
    return bitcast<f32>(((m >> 3u) + 120u) << 23u | (m & 7u) << 20u);
}

fn e4m3_round(y: f32) -> f32 {
    return e4m3_value(e4m3_magnitude(y));
}

// |v| rounded to E2M1 (nearest, ties to even, saturating at 6).
fn e2m1_magnitude(a: f32) -> f32 {
    if (a <= 0.25) { return 0.0; }
    if (a < 0.75) { return 0.5; }
    if (a <= 1.25) { return 1.0; }
    if (a < 1.75) { return 1.5; }
    if (a <= 2.5) { return 2.0; }
    if (a < 3.5) { return 3.0; }
    if (a <= 5.0) { return 4.0; }
    return 6.0;
}

// The reconstruction of `xv` through block scale `s` (inverse `inv`): `(e2m1(xv * inv) * s) * outer`, signed.
fn nv_recon(xv: f32, s: f32, inv: f32, outer: f32) -> f32 {
    if (s == 0.0) { return 0.0; }
    let v = xv * inv;
    let m = (e2m1_magnitude(abs(v)) * s) * outer;
    return select(m, -m, v < 0.0);
}

// Pairwise-tree sum of |recon - x| over the 16 values of the block at `base` for one candidate scale.
fn nv_error(base: u32, s: f32, inv: f32, outer: f32) -> f32 {
    var e: array<f32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        let xv = x[base + i];
        e[i] = abs(nv_recon(xv, s, inv, outer) - xv);
    }
    for (var w = 8u; w >= 1u; w = w / 2u) {
        for (var i = 0u; i < w; i = i + 1u) {
            e[i] = e[2u * i] + e[2u * i + 1u];
        }
    }
    return e[0];
}

fn nv_inverse(s: f32, dec: f32) -> f32 {
    if (s == 0.0) { return 0.0; }
    return 1.0 / (s * dec);
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    if (row >= p.rows) { return; }
    let base = row * p.k;
    // The row amax (read by the NVFP4 scheme only).
    var acc = 0.0;
    for (var c = t; c < p.k; c = c + 64u) {
        acc = max(acc, abs(x[base + c]));
    }
    partial[t] = acc;
    workgroupBarrier();
    var amax = 0.0;
    for (var i = 0u; i < 64u; i = i + 1u) {
        amax = max(amax, partial[i]);
    }
    if (p.scheme == 0u) {
        let tiny = amax < TINY;
        let enc = 1536.0 / max(amax, TINY);
        let dec = 1.0 / enc;
        let outer = amax * (1.0 / 1536.0);
        let enc6 = enc * (1.0 / 6.0);
        let enc4 = enc * 0.25;
        for (var b = t; b < p.k / 16u; b = b + 64u) {
            let o = base + b * 16u;
            if (tiny) {
                for (var i = 0u; i < 16u; i = i + 1u) { out[o + i] = 0.0; }
                continue;
            }
            var bmax = 0.0;
            for (var i = 0u; i < 16u; i = i + 1u) { bmax = max(bmax, abs(x[o + i])); }
            let s6 = e4m3_round(min(bmax * enc6, 448.0));
            let s4 = e4m3_round(min(bmax * enc4, 448.0));
            let inv6 = nv_inverse(s6, dec);
            let inv4 = nv_inverse(s4, dec);
            let four = nv_error(o, s4, inv4, outer) < nv_error(o, s6, inv6, outer);
            let s = select(s6, s4, four);
            let inv = select(inv6, inv4, four);
            for (var i = 0u; i < 16u; i = i + 1u) {
                out[o + i] = nv_recon(x[o + i], s, inv, outer);
            }
        }
    } else {
        for (var b = t; b < p.k / 32u; b = b + 64u) {
            let o = base + b * 32u;
            var bmax = 0.0;
            for (var i = 0u; i < 32u; i = i + 1u) { bmax = max(bmax, abs(x[o + i])); }
            let raw = bmax * (1.0 / 448.0);
            if (raw < MIN_NORMAL) {
                for (var i = 0u; i < 32u; i = i + 1u) { out[o + i] = 0.0; }
                continue;
            }
            let e = min(((bitcast<u32>(raw) + 0x7FFFFFu) & 0x7F800000u) >> 23u, 253u);
            let scale = bitcast<f32>(e << 23u);
            let inverse = bitcast<f32>((254u - e) << 23u);
            for (var i = 0u; i < 32u; i = i + 1u) {
                let v = x[o + i] * inverse;
                let m = e4m3_round(min(abs(v), 448.0)) * scale;
                out[o + i] = select(m, -m, v < 0.0);
            }
        }
    }
}

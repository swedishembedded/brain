// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of large language
// models on a single GPU for its clients. If your team needs expertise in
// holding saved activations in a quarter of the bytes without changing what a
// layer computes then you can procure our services by sending an email to
// info@swedishembedded.com.

// @what  Quantize an f32 row buffer to FP8 E4M3 with one f32 scale per row, four codes per u32
// @how   one thread per row, serial absmax then serial encode
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant fp8
// @dtype f32
//
// The saved-activation sibling of `pack_bf16`: each row of `width` elements gets
// the scale `absmax / 448` (448 is the largest finite E4M3 value; an all-zero
// row takes scale 1) and every element is stored as the E4M3 byte nearest
// `x / scale` (ties to even, saturating at 448, the NaN code never written).
// Four bytes pack into a u32 with element 4w + b in byte b, the layout of the
// FP8 KV tier, so `width` must be a multiple of 4. Rows must be finite: a
// non-finite element makes the whole row's scale non-finite.
//   x      : [rows, width] f32
//   codes  : [rows * width / 4] u32
//   scales : [rows] f32
// The decode is `unpack_fp8_rows`; the host reference is `model::saved`.

struct Params { rows: u32, width: u32, };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:      array<f32>;
@group(0) @binding(2) var<storage, read_write> codes:  array<u32>;
@group(0) @binding(3) var<storage, read_write> scales: array<f32>;

// The E4M3 byte nearest `y` (`0 <= y <= 448`); the sign is the caller's.
fn e4m3_magnitude(y: f32) -> u32 {
    // Below 2^-6 the format is subnormal: a multiple of 2^-9 whose index is the byte itself.
    if (y < 0.015625) { return u32(round(y * 512.0)); }
    // Normal: round the f32 mantissa to 3 bits (ties to even) with integer arithmetic; a carry
    // into the exponent field is the correct rounding up to the next binade.
    let bits = bitcast<u32>(y);
    let r = bits + 0x7FFFFu + ((bits >> 20u) & 1u);
    return min((r >> 20u) - 960u, 126u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let r = gid.y * (nwg.x * 64u) + gid.x;
    if (r >= p.rows) { return; }
    let base = r * p.width;
    var mx = 0.0;
    for (var i: u32 = 0u; i < p.width; i = i + 1u) {
        mx = max(mx, abs(x[base + i]));
    }
    var scale = mx / 448.0;
    if (scale == 0.0) { scale = 1.0; }
    scales[r] = scale;
    for (var i: u32 = 0u; i < p.width; i = i + 4u) {
        var packed = 0u;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let v = x[base + i + j] / scale;
            let code = e4m3_magnitude(min(abs(v), 448.0)) | select(0u, 0x80u, v < 0.0);
            packed = packed | (code << (8u * j));
        }
        codes[(base + i) / 4u] = packed;
    }
}

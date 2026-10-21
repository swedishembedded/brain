// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of large language
// models on a single GPU for its clients. If your team needs expertise in
// holding saved activations in a quarter of the bytes without changing what a
// layer computes then you can procure our services by sending an email to
// info@swedishembedded.com.

// @what  Decode FP8 E4M3 rows with a per-row f32 scale (the inverse of pack_fp8_rows) to f32
// @how   one thread per output element; integer bit decode of the byte, one multiply
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant fp8
// @dtype f32
//
// out[i] = e4m3(byte i of codes) * scales[i / width]. Exact: the E4M3 value and
// the scale multiply in f32, so the round trip is the one-way rounding
// `pack_fp8_rows` applied.
//   codes  : [rows * width / 4] u32
//   scales : [rows] f32
//   out    : [rows, width] f32

struct Params { total: u32, width: u32, };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       codes:  array<u32>;
@group(0) @binding(2) var<storage, read>       scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> out:    array<f32>;

fn fp8_e4m3(b: u32) -> f32 {
    let normal = bitcast<f32>(((b & 127u) << 20u) + 1006632960u);
    let sub = f32(b & 7u) * 0.001953125;
    let mag = select(normal, sub, (b & 120u) == 0u);
    return bitcast<f32>(bitcast<u32>(mag) | ((b & 128u) << 24u));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    // 2D-grid safe linear thread index (identity for 1D dispatch).
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.total) { return; }
    let byte = (codes[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu;
    out[i] = fp8_e4m3(byte) * scales[i / p.width];
}

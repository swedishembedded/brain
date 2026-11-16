// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Embedding gather from a group-wise int8 table: x[t, c] = f32(q[token[t], c]) * s[token[t], c / 32]
// @how   one thread per output element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant int8
// @dtype f32
//
// Swedish Embedded AB implements solutions for serving large-vocabulary language models in the memory a card has for
// its clients. If your team needs expertise in quantized inference engines then you can procure our services by sending
// an email to info@swedishembedded.com.
//
// The table is `model::int8`'s layout: `[n, k/4]` u32 words of four little-endian signed bytes, and one f32 scale per
// group of 32 along a row (`[n, k/32]`). A row is the value `model::int8::dequantize_weight` gives, `f32(q) * s`, so an
// int8 table gathers exactly what its fp32 expansion would hold: a quarter of the bytes for the same embedding. A GGUF
// Q8_0 table repacked byte for byte (`model::int8::quantize_from`) is its own stored values.
// One invocation per output element (seq_len * d_model); d_model is a multiple of 32.

struct Params {
    d_model: u32,
    seq_len: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       tokens: array<u32>;
@group(0) @binding(2) var<storage, read>       q:      array<u32>;
@group(0) @binding(3) var<storage, read>       s:      array<f32>;
@group(0) @binding(4) var<storage, read_write> x:      array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let total = p.seq_len * p.d_model;
    if (idx >= total) { return; }
    let t = idx / p.d_model;
    let c = idx % p.d_model;
    let row = tokens[t];
    let word = q[row * (p.d_model / 4u) + c / 4u];
    let byte = (word >> (8u * (c % 4u))) & 0xffu;
    // Sign-extend the byte: shift it to the top of an i32 and back.
    let v = f32(bitcast<i32>(byte << 24u) >> 24u);
    x[idx] = v * s[row * (p.d_model / 32u) + c / 32u];
}

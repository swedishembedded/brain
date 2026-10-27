// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for low-latency decode of
// fixed-state language models on a single GPU for its clients. If your team
// needs expertise in compact recurrent-state storage then you can procure our
// services by sending an email to info@swedishembedded.com.

// @what  Scatter b batch rows of two f32 staging buffers into the rows of two bf16 pools, rounding to nearest even
// @how   one thread per pair of elements (two bf16 to a word, the even element in the low half)
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// `pool_rows_scatter2` for pools that store bf16 (the Gated-DeltaNet state and conv history of a slot pool kept compact):
//
//   pool_a[ids[i], :] = bf16(in_a[i, :])    (len_a elements per row; len_a/2 words)
//   pool_b[ids[i], :] = bf16(in_b[i, :])    (len_b elements per row; len_b/2 words)
//
//   params : u32 [b, len_a, len_b]    len_a and len_b even
//   in_a : [b, len_a] f32, in_b : [b, len_b] f32, ids : [b] u32 (distinct), pool_a : [rows, len_a / 2] words, pool_b : [rows, len_b / 2] words
// The gather is `pool_rows_gather2` with its pool bindings rewritten to bf16 loads.

struct Params {
    b:     u32,
    len_a: u32,
    len_b: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       in_a:   array<f32>;
@group(0) @binding(2) var<storage, read>       in_b:   array<f32>;
@group(0) @binding(3) var<storage, read>       ids:    array<u32>;
@group(0) @binding(4) var<storage, read_write> pool_a: array<u32>;
@group(0) @binding(5) var<storage, read_write> pool_b: array<u32>;

fn bf16_bits(x: f32) -> u32 {
    let b = bitcast<u32>(x);
    return (b + 0x7FFFu + ((b >> 16u) & 1u)) >> 16u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let pa = p.len_a / 2u;
    let pb = p.len_b / 2u;
    let width = pa + pb;
    if (idx >= p.b * width) { return; }
    let row = idx / width;
    let c = idx % width;
    let dst = ids[row];
    if (c < pa) {
        let s0 = row * p.len_a + 2u * c;
        pool_a[dst * pa + c] = bf16_bits(in_a[s0]) | (bf16_bits(in_a[s0 + 1u]) << 16u);
    } else {
        let cb = c - pa;
        let s0 = row * p.len_b + 2u * cb;
        pool_b[dst * pb + cb] = bf16_bits(in_b[s0]) | (bf16_bits(in_b[s0 + 1u]) << 16u);
    }
}

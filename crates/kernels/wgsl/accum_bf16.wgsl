// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements low-precision gradient accumulation for
// memory-bounded training for its clients. If your team needs expertise in
// training with bf16 gradients without a full-precision gradient copy, you
// can procure our services by sending an email to info@swedishembedded.com.

// @what  Add an f32 tile into a packed bf16 gradient buffer (round to nearest even) and clear the tile
// @how   one thread per packed word (two elements), so no two threads share a word
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// The accumulation step of a weight gradient stored in bf16: `dw[elem0 + i] = bf16(f32(dw[elem0 + i]) + t[i])`
// for the `2 * words` elements of the tile, where `dw` is the flat bf16 gradient (two elements per word, low
// half first, `model::half::pack_bf16`'s layout) and `elem0` is even. The tile `t` is zeroed afterwards, so
// the next product can accumulate into the same scratch without a separate clear.
//
//   params: words, elem0

struct Params {
    words: u32,
    elem0: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> t:  array<f32>;
@group(0) @binding(2) var<storage, read_write> dw: array<u32>;

fn to_bf16(v: f32) -> u32 {
    let b = bitcast<u32>(v);
    if (v != v) { return (b >> 16u) | 0x40u; }
    return (b + 0x7FFFu + ((b >> 16u) & 1u)) >> 16u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.words) { return; }
    let wi = (p.elem0 >> 1u) + i;
    let old = dw[wi];
    let lo = bitcast<f32>((old & 0xFFFFu) << 16u) + t[2u * i];
    let hi = bitcast<f32>(old & 0xFFFF0000u) + t[2u * i + 1u];
    dw[wi] = to_bf16(lo) | (to_bf16(hi) << 16u);
    t[2u * i] = 0.0;
    t[2u * i + 1u] = 0.0;
}

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving long-context models at
// a fixed memory budget for its clients. If your team needs expertise in KV
// cache management and sliding-window rings then you can procure our services
// by sending an email to info@swedishembedded.com.

// @what  pack_bf16 with offsets: out[dst_word + j] = bf16(x[src + 2j]) | bf16(x[src + 2j + 1]) << 16
// @how   one thread per output word; round-to-nearest-even on the 16 dropped bits, NaN stays NaN
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// `pack_bf16.wgsl` writing `total` elements of `x` starting at element `src`
// into `out` starting at word `dst_word`. The offsets are uniform parameters
// because a bf16 ring's row offset is not a multiple of the device's
// `min_storage_buffer_offset_alignment`, so it cannot be a binding offset.
// `total` is even (a ring row is).

struct Params {
    total:    u32,
    src:      u32,
    dst_word: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x:   array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

fn bf16_bits(v: f32) -> u32 {
    let b = bitcast<u32>(v);
    if ((b & 0x7fffffffu) > 0x7f800000u) {
        return (b >> 16u) | 0x0040u;
    }
    return (b + 0x7fffu + ((b >> 16u) & 1u)) >> 16u;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let j = gid.y * (nwg.x * 64u) + gid.x;
    if (j >= (p.total + 1u) / 2u) { return; }
    let lo = bf16_bits(x[p.src + 2u * j]);
    var hi = 0u;
    if (2u * j + 1u < p.total) { hi = bf16_bits(x[p.src + 2u * j + 1u]); }
    out[p.dst_word + j] = lo | (hi << 16u);
}

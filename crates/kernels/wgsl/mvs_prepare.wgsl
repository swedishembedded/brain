// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  MVS image level: packed RGB8 photo -> exact box-halved gray + RGB planes
// @how   one thread per output pixel, serial loop over its factor x factor source block
// @opt   2
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// One pyramid level of a photograph for multi-view stereo: output pixel (x, y)
// is the mean of the source block [x f, (x+1) f) x [y f, (y+1) f), f = 2^h -
// exactly h successive 2x2 halvings, so with continuous pixel coordinates
// (pixel i's centre at i + 0.5) the level's intrinsics are the photograph's
// scaled by exactly 1/f. A source dimension that is not a multiple of f loses
// its last partial block, which keeps that scaling exact.
//
// `src` is the photograph as interleaved 8-bit RGB, four bytes to a word,
// little-endian (byte b of the file is bits 8(b%4).. of word b/4).
// `out` gets four planes of `plane` floats: gray (Rec. 601 luma), R, G, B,
// all in [0, 1].

struct Params {
    src_w: u32,
    src_h: u32,
    w: u32,
    h: u32,
    factor: u32,
    plane: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       src: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

fn byte_at(b: u32) -> f32 {
    return f32((src[b >> 2u] >> (8u * (b & 3u))) & 255u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.w * p.h) { return; }
    let x = i % p.w;
    let y = i / p.w;
    var rgb = vec3<f32>(0.0, 0.0, 0.0);
    for (var dy = 0u; dy < p.factor; dy = dy + 1u) {
        let row = (y * p.factor + dy) * p.src_w;
        for (var dx = 0u; dx < p.factor; dx = dx + 1u) {
            let b = (row + x * p.factor + dx) * 3u;
            rgb = rgb + vec3<f32>(byte_at(b), byte_at(b + 1u), byte_at(b + 2u));
        }
    }
    rgb = rgb / (255.0 * f32(p.factor * p.factor));
    out[i] = dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
    out[p.plane + i] = rgb.x;
    out[2u * p.plane + i] = rgb.y;
    out[3u * p.plane + i] = rgb.z;
}

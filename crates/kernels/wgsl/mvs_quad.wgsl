// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  MVS bilinear quads: a gray plane -> each pixel's 2x2 neighbourhood as four 16-bit unorms
// @how   one thread per pixel
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// PatchMatch samples every source view at hundreds of warped, non-integer
// positions per pixel and hypothesis, and each bilinear sample is four
// gathers - measured as half of `mvs_pm`'s time, and 1.23x faster as one
// 8-byte load (the finest level of a 1632x1224 capture, with `mvs_pm`'s
// views-outermost loop order; with hypotheses outermost it gained
// nothing). Here each pixel
// (x, y) stores the four taps a bilinear sample anchored at it needs,
// g(x, y), g(x+1, y), g(x, y+1), g(x+1, y+1) (clamped at the border), as
// 16-bit unorms in two words, so one 8-byte load serves a sample. This is
// storage precision only: the values are decoded to f32 before any
// arithmetic, and 16 bits exceed the 8-bit photographs they come from.
//
// quad[i] = (g(x, y) | g(x+1, y) << 16, g(x, y+1) | g(x+1, y+1) << 16).

struct Params {
    w: u32,
    h: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       gray: array<f32>;
@group(0) @binding(2) var<storage, read_write> quad: array<vec2<u32>>;

fn unorm16(g: f32) -> u32 {
    return u32(round(clamp(g, 0.0, 1.0) * 65535.0));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.w * p.h) { return; }
    let x = i % p.w;
    let y = i / p.w;
    let x1 = min(x + 1u, p.w - 1u);
    let y1 = min(y + 1u, p.h - 1u);
    let top = unorm16(gray[y * p.w + x]) | (unorm16(gray[y * p.w + x1]) << 16u);
    let bot = unorm16(gray[y1 * p.w + x]) | (unorm16(gray[y1 * p.w + x1]) << 16u);
    quad[i] = vec2<u32>(top, bot);
}

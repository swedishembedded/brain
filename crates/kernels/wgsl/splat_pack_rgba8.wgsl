// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Pack the RGBA f32 framebuffer into one u32 (r / g<<8 / b<<16 / a<<24) per pixel - quarters the demo readback bandwidth
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Pack the RGBA f32 framebuffer into one u32 (r | g<<8 | b<<16 | a<<24) per
// pixel — quarters the demo readback bandwidth. Manual packing (no
// pack4x8unorm: the CPU JIT does not implement it).

struct Params {
    n: u32, // W*H
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       img: array<f32>; // n*4
@group(0) @binding(2) var<storage, read_write> out: array<u32>; // n

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.y * (nwg.x * 64u) + gid.x;
    if (i >= p.n) { return; }
    let r = u32(clamp(img[i * 4u], 0.0, 1.0) * 255.0 + 0.5);
    let g = u32(clamp(img[i * 4u + 1u], 0.0, 1.0) * 255.0 + 0.5);
    let b = u32(clamp(img[i * 4u + 2u], 0.0, 1.0) * 255.0 + 0.5);
    let a = u32(clamp(img[i * 4u + 3u], 0.0, 1.0) * 255.0 + 0.5);
    out[i] = r | (g << 8u) | (b << 16u) | (a << 24u);
}

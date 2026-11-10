// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training depthwise-convolution kernel gradient: dk[ky*kw+kx, c] += sum over outputs of x * gy
// @how   one thread per kernel element, looping over the output positions; accumulates
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Geometry as the forward `mirage_conv2d` / `mirage_dwconv`: zero padding k/2, output (h + 2 (k/2) - k) / s + 1.
// Params: h, w, c, ho, wo, kh, kw, sy, sx. Dispatch: kh*kw*c threads.

struct Params { h: u32, w: u32, c: u32, ho: u32, wo: u32, kh: u32, kw: u32, sy: u32, sx: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>  x: array<f32>;
@group(0) @binding(2) var<storage, read>  gy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dk: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.kh * p.kw * p.c) { return; }
    let ch = idx % p.c;
    let tap = idx / p.c;
    let ky = tap / p.kw;
    let kx = tap % p.kw;
    var acc = 0.0;
    for (var oy = 0u; oy < p.ho; oy = oy + 1u) {
        let iy = i32(oy * p.sy + ky) - i32(p.kh / 2u);
        if (iy < 0 || iy >= i32(p.h)) { continue; }
        for (var ox = 0u; ox < p.wo; ox = ox + 1u) {
            let ix = i32(ox * p.sx + kx) - i32(p.kw / 2u);
            if (ix < 0 || ix >= i32(p.w)) { continue; }
            acc = acc + x[(u32(iy) * p.w + u32(ix)) * p.c + ch] * gy[(oy * p.wo + ox) * p.c + ch];
        }
    }
    dk[idx] = dk[idx] + acc;
}

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training depthwise-convolution input gradient: dx[iy, ix, c] += sum of gy * kernel over the taps that read it
// @how   one thread per input element, looping over the kernel taps; accumulates
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Geometry as the forward `mirage_conv2d` / `mirage_dwconv`: zero padding k/2, output (h + 2 (k/2) - k) / s + 1.
// Params: h, w, c, ho, wo, kh, kw, sy, sx. Dispatch: h*w*c threads.

struct Params { h: u32, w: u32, c: u32, ho: u32, wo: u32, kh: u32, kw: u32, sy: u32, sx: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>  gy: array<f32>;
@group(0) @binding(2) var<storage, read>  k: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.h * p.w * p.c) { return; }
    let ch = idx % p.c;
    let pix = idx / p.c;
    let ix = i32(pix % p.w);
    let iy = i32(pix / p.w);
    var acc = 0.0;
    for (var ky = 0u; ky < p.kh; ky = ky + 1u) {
        let ny = iy + i32(p.kh / 2u) - i32(ky);
        if (ny < 0 || ny % i32(p.sy) != 0) { continue; }
        let oy = ny / i32(p.sy);
        if (oy >= i32(p.ho)) { continue; }
        for (var kx = 0u; kx < p.kw; kx = kx + 1u) {
            let nx = ix + i32(p.kw / 2u) - i32(kx);
            if (nx < 0 || nx % i32(p.sx) != 0) { continue; }
            let ox = nx / i32(p.sx);
            if (ox >= i32(p.wo)) { continue; }
            acc = acc + gy[(u32(oy) * p.wo + u32(ox)) * p.c + ch] * k[(ky * p.kw + kx) * p.c + ch];
        }
    }
    dx[idx] = dx[idx] + acc;
}

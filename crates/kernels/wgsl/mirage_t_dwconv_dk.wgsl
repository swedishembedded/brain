// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Training depthwise-convolution kernel gradient: dk[ky*kw+kx, c] += sum over outputs of x * gy
// @how   64-thread workgroup per 8 kernel elements (8 output lanes), 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Geometry as the forward `mirage_conv2d` / `mirage_dwconv`: zero padding k/2, output (h + 2 (k/2) - k) / s + 1.
// Tiled column reduction: a 64-thread workgroup owns 8 adjacent columns; thread (rl, cl) sums rows rl, rl+8, ... of
// column cl, then the 8 row lanes are folded in a fixed order (one barrier), so the sum is deterministic and the
// loads of a row lane touch 8 adjacent columns.
// Params: h, w, c, ho, wo, kh, kw, sy, sx. Dispatch: ceil(kh*kw*c / 8) workgroups.

struct Params { h: u32, w: u32, c: u32, ho: u32, wo: u32, kh: u32, kw: u32, sy: u32, sx: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>  x: array<f32>;
@group(0) @binding(2) var<storage, read>  gy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dk: array<f32>;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) li: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg.y * nwg.x + wg.x;
    let t = li.x;
    let col0 = row * 8u;
    let cl = t % 8u;
    let rl = t / 8u;
    let c = col0 + cl;
    var s = 0.0;
    if (c < p.kh * p.kw * p.c) {
        let ch = c % p.c;
        let tap = c / p.c;
        let ky = tap / p.kw;
        let kx = tap % p.kw;
        for (var pos = rl; pos < p.ho * p.wo; pos = pos + 8u) {
            let oy = pos / p.wo;
            let ox = pos % p.wo;
            let iy = i32(oy * p.sy + ky) - i32(p.kh / 2u);
            let ix = i32(ox * p.sx + kx) - i32(p.kw / 2u);
            if (iy >= 0 && iy < i32(p.h) && ix >= 0 && ix < i32(p.w)) {
                s = s + x[(u32(iy) * p.w + u32(ix)) * p.c + ch] * gy[pos * p.c + ch];
            }
        }
    }
    partial[t] = s;
    workgroupBarrier();
    if (t < 8u) {
        let c2 = col0 + t;
        if (c2 < p.kh * p.kw * p.c) {
            var tot = 0.0;
            for (var k = 0u; k < 8u; k = k + 1u) { tot = tot + partial[k * 8u + t]; }
            dk[c2] = dk[c2] + tot;
        }
    }
}

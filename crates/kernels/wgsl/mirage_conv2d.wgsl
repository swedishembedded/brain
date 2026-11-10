// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Dense NHWC convolution, HWIO weights, zero padding k/2, any stride (Mirage stem and reader first layers)
// @how   one thread per output element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// y[b, oy, ox, o] = sum_{ky,kx,i} x[b, oy*sy + ky - kh/2, ox*sx + kx - kw/2, i] * w[ky, kx, i, o]
// Params: batch, h, w, cin, ho, wo, cout, kh, kw, sy, sx. Dispatch: batch*ho*wo*cout threads.

struct Params { batch: u32, h: u32, w: u32, cin: u32, ho: u32, wo: u32, cout: u32, kh: u32, kw: u32, sy: u32, sx: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read>       w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    if (idx >= p.batch * p.ho * p.wo * p.cout) { return; }
    let o = idx % p.cout;
    let pix = idx / p.cout;
    let ox = pix % p.wo;
    let oy = (pix / p.wo) % p.ho;
    let b = pix / (p.wo * p.ho);
    var acc = 0.0;
    for (var ky = 0u; ky < p.kh; ky = ky + 1u) {
        let iy = i32(oy * p.sy + ky) - i32(p.kh / 2u);
        if (iy < 0 || iy >= i32(p.h)) { continue; }
        for (var kx = 0u; kx < p.kw; kx = kx + 1u) {
            let ix = i32(ox * p.sx + kx) - i32(p.kw / 2u);
            if (ix < 0 || ix >= i32(p.w)) { continue; }
            let src = ((b * p.h + u32(iy)) * p.w + u32(ix)) * p.cin;
            let wk = (ky * p.kw + kx) * p.cin * p.cout + o;
            for (var i = 0u; i < p.cin; i = i + 1u) {
                acc = acc + x[src + i] * w[wk + i * p.cout];
            }
        }
    }
    y[idx] = acc;
}

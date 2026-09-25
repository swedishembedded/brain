// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  bilateral grid gradient of the photometric camera model: dLoss/d(3x4 affine cell) gathered over the pixels each cell's trilinear stencil reaches
// @how   64-thread workgroup tile, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
// @import isp
//
// One workgroup per grid cell (i, j, k), a gather with no atomics: the cell's
// stencil reaches the pixels whose continuous grid coordinate lies within one
// cell of it in x and y, and of those, the ones whose guide luma lies within
// one cell of k. Each thread walks a strided share of that rectangle, recomputes
// the pixel's global camera output e (`isp_global`), and accumulates
//   w * dLoss/d(prediction) (x) [e, 1]
// with w the cell's trilinear weight - the hat function max(0, 1 - |u - i|) per
// axis, which is exactly the weight `isp_pixel` slices with. The guide's own
// dependence on e is `isp_pixel`'s business (it moves e, not the cell).
//
// Runs BEFORE `isp_pixel`'s backward, which overwrites the upstream in place.
// Each pixel is read by the 4 x-y cells around it at up to 8 luma levels.

struct Params {
    v: IspView,
    c: IspCam,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       img:   array<f32>; // W*H*4 radiance
@group(0) @binding(2) var<storage, read>       up:    array<f32>; // W*H*4 dLoss/d(prediction)
@group(0) @binding(3) var<storage, read_write> dgrid: array<f32>; // gx*gy*gz*12

var<workgroup> acc: array<f32, 768>; // 12 gradients x 64 threads

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let t = lid.x;
    let cell = wid.y * nwg.x + wid.x;
    let n_cells = p.v.gx * p.v.gy * p.v.gz;
    var g0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var g1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var g2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if (cell < n_cells) {
        let ci = cell % p.v.gx;
        let cj = (cell / p.v.gx) % p.v.gy;
        let ck = cell / (p.v.gx * p.v.gy);
        // the stencil's pixel rectangle, a pixel of margin either side (the
        // hat weight is exact; the rectangle only has to contain its support)
        let sx = f32(p.v.width) / f32(p.v.gx - 1u);
        let sy = f32(p.v.height) / f32(p.v.gy - 1u);
        let x_lo = u32(max(floor((f32(ci) - 1.0) * sx) - 1.0, 0.0));
        let x_hi = min(u32(max(ceil((f32(ci) + 1.0) * sx) + 1.0, 0.0)), p.v.width);
        let y_lo = u32(max(floor((f32(cj) - 1.0) * sy) - 1.0, 0.0));
        let y_hi = min(u32(max(ceil((f32(cj) + 1.0) * sy) + 1.0, 0.0)), p.v.height);
        let nx = x_hi - x_lo;
        let n = nx * (y_hi - y_lo);
        for (var q = t; q < n; q = q + 64u) {
            let x = x_lo + q % nx;
            let y = y_lo + q / nx;
            let uxy = isp_grid_coord(p.v, x, y, 0.0);
            let wxy = max(1.0 - abs(uxy.x - f32(ci)), 0.0) * max(1.0 - abs(uxy.y - f32(cj)), 0.0);
            if (wxy <= 0.0) { continue; }
            let pix = y * p.v.width + x;
            let l = vec3<f32>(img[pix * 4u], img[pix * 4u + 1u], img[pix * 4u + 2u]);
            let tr = isp_global(p.v, p.c, l, isp_r2(p.v, x, y));
            let uz = clamp(isp_luma(tr.e), 0.0, 1.0) * f32(p.v.gz - 1u);
            let w = wxy * max(1.0 - abs(uz - f32(ck)), 0.0);
            if (w <= 0.0) { continue; }
            let e4 = vec4<f32>(tr.e, 1.0);
            g0 = g0 + (w * up[pix * 4u]) * e4;
            g1 = g1 + (w * up[pix * 4u + 1u]) * e4;
            g2 = g2 + (w * up[pix * 4u + 2u]) * e4;
        }
    }
    acc[t * 12u] = g0.x;
    acc[t * 12u + 1u] = g0.y;
    acc[t * 12u + 2u] = g0.z;
    acc[t * 12u + 3u] = g0.w;
    acc[t * 12u + 4u] = g1.x;
    acc[t * 12u + 5u] = g1.y;
    acc[t * 12u + 6u] = g1.z;
    acc[t * 12u + 7u] = g1.w;
    acc[t * 12u + 8u] = g2.x;
    acc[t * 12u + 9u] = g2.y;
    acc[t * 12u + 10u] = g2.z;
    acc[t * 12u + 11u] = g2.w;
    workgroupBarrier();
    if (t < 12u && cell < n_cells) {
        var sum = 0.0;
        for (var j = 0u; j < 64u; j = j + 1u) {
            sum = sum + acc[j * 12u + t];
        }
        dgrid[cell * 12u + t] = sum;
    }
}

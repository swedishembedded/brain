// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  photometric camera model per pixel: radiance -> recorded value (forward), or its adjoint with per-workgroup camera-parameter gradient partials (backward)
// @how   64-thread workgroup tile, 1 barrier
// @opt   4
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
// @import isp
//
// One invocation per pixel of a view, through the chain `lib/isp.wgsl`
// describes. `p.v.mode` selects the direction:
//
// * 0, forward: `io` receives the prediction (alpha 0) the loss compares with
//   the photograph.
// * 1, backward: `io` holds dLoss/d(prediction) and is overwritten in place
//   with dLoss/d(radiance) (alpha untouched, it belongs to the geometry
//   terms); every workgroup writes the sum over its pixels of dLoss/d(camera)
//   in the order the host chains it to the fit's parameters:
//     [0, 3)   d gain, per channel
//     [3, 12)  d vignetting coefficient a_k of channel c at 3 + 3 k + c
//     [12, 21) d colour correction entry (r, c) at 12 + 3 r + c
//     [21, 45) d response slope of segment j, channel c at 21 + 3 j + c
//   `dw_splitk_reduce` folds the workgroups.
//
// The bilateral grid's own gradient needs the prediction's upstream before
// this kernel replaces it, so `isp_grid_grad` runs first.

struct Params {
    v: IspView,
    c: IspCam,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       img:     array<f32>; // W*H*4 radiance (renderer output)
@group(0) @binding(2) var<storage, read>       grid:    array<f32>; // gx*gy*gz*12, deviation from identity
@group(0) @binding(3) var<storage, read_write> io:      array<f32>; // W*H*4 prediction | upstream -> d radiance
@group(0) @binding(4) var<storage, read_write> partial: array<f32>; // n_wg*45

var<workgroup> acc: array<f32, 2880>; // 45 gradients x 64 threads

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let pix = gid.y * (nwg.x * 64u) + gid.x;
    let t = lid.x;
    let base = t * 45u;
    for (var k = 0u; k < 45u; k = k + 1u) {
        acc[base + k] = 0.0;
    }
    if (pix < p.v.width * p.v.height) {
        let x = pix % p.v.width;
        let y = pix / p.v.width;
        let r2 = isp_r2(p.v, x, y);
        let l = vec3<f32>(img[pix * 4u], img[pix * 4u + 1u], img[pix * 4u + 2u]);
        let tr = isp_global(p.v, p.c, l, r2);
        let e4 = vec4<f32>(tr.e, 1.0);
        var dp = vec3<f32>(0.0, 0.0, 0.0);
        if (p.v.mode == 1u) {
            dp = vec3<f32>(io[pix * 4u], io[pix * 4u + 1u], io[pix * 4u + 2u]);
        }
        var out = tr.e;
        var de = dp;
        if (p.v.has_grid != 0u) {
            // slice the grid (D) and its derivative along the luma axis (Z)
            let luma = isp_luma(tr.e);
            let u = isp_grid_coord(p.v, x, y, luma);
            let i0 = isp_grid_base(u.x, p.v.gx);
            let j0 = isp_grid_base(u.y, p.v.gy);
            let k0 = isp_grid_base(u.z, p.v.gz);
            let f = u - vec3<f32>(f32(i0), f32(j0), f32(k0));
            var d0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            var d1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            var d2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            var z0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            var z1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            var z2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            for (var q = 0u; q < 8u; q = q + 1u) {
                let bx = q & 1u;
                let by = (q >> 1u) & 1u;
                let bz = q >> 2u;
                var wx = 1.0 - f.x;
                if (bx == 1u) { wx = f.x; }
                var wy = 1.0 - f.y;
                if (by == 1u) { wy = f.y; }
                var wz = 1.0 - f.z;
                var sz = -1.0;
                if (bz == 1u) { wz = f.z; sz = 1.0; }
                let o = (((k0 + bz) * p.v.gy + j0 + by) * p.v.gx + i0 + bx) * 12u;
                let r0 = vec4<f32>(grid[o], grid[o + 1u], grid[o + 2u], grid[o + 3u]);
                let r1 = vec4<f32>(grid[o + 4u], grid[o + 5u], grid[o + 6u], grid[o + 7u]);
                let rr = vec4<f32>(grid[o + 8u], grid[o + 9u], grid[o + 10u], grid[o + 11u]);
                let wxy = wx * wy;
                d0 = d0 + (wxy * wz) * r0;
                d1 = d1 + (wxy * wz) * r1;
                d2 = d2 + (wxy * wz) * rr;
                z0 = z0 + (wxy * sz) * r0;
                z1 = z1 + (wxy * sz) * r1;
                z2 = z2 + (wxy * sz) * rr;
            }
            out = tr.e + vec3<f32>(dot(d0, e4), dot(d1, e4), dot(d2, e4));
            de = dp + vec3<f32>(
                d0.x * dp.x + d1.x * dp.y + d2.x * dp.z,
                d0.y * dp.x + d1.y * dp.y + d2.y * dp.z,
                d0.z * dp.x + d1.z * dp.y + d2.z * dp.z);
            if (luma > 0.0 && luma < 1.0) {
                let g = (dp.x * dot(z0, e4) + dp.y * dot(z1, e4) + dp.z * dot(z2, e4)) * f32(p.v.gz - 1u);
                de = de + g * vec3<f32>(0.2126, 0.7152, 0.0722);
            }
        }
        if (p.v.mode == 0u) {
            io[pix * 4u] = out.x;
            io[pix * 4u + 1u] = out.y;
            io[pix * 4u + 2u] = out.z;
            io[pix * 4u + 3u] = 0.0;
        } else {
            var dy = de;
            if (p.v.encode != 0u) {
                dy = de * vec3<f32>(isp_oetf_grad(tr.y.x), isp_oetf_grad(tr.y.y), isp_oetf_grad(tr.y.z));
            }
            for (var j = 0u; j < 8u; j = j + 1u) {
                let g = dy * isp_portion(j, tr.m);
                acc[base + 21u + 3u * j] = g.x;
                acc[base + 22u + 3u * j] = g.y;
                acc[base + 23u + 3u * j] = g.z;
            }
            let dm = dy * isp_crf_slope(p.c, tr.m);
            let m0 = p.c.m0.xyz;
            let m1 = p.c.m1.xyz;
            let m2 = p.c.m2.xyz;
            let dx = m0 * dm.x + m1 * dm.y + m2 * dm.z;
            acc[base + 12u] = dm.x * tr.x.x;
            acc[base + 13u] = dm.x * tr.x.y;
            acc[base + 14u] = dm.x * tr.x.z;
            acc[base + 15u] = dm.y * tr.x.x;
            acc[base + 16u] = dm.y * tr.x.y;
            acc[base + 17u] = dm.y * tr.x.z;
            acc[base + 18u] = dm.z * tr.x.x;
            acc[base + 19u] = dm.z * tr.x.y;
            acc[base + 20u] = dm.z * tr.x.z;
            let gain = p.c.gain.xyz;
            let dg = dx * tr.vig * l;
            acc[base] = dg.x;
            acc[base + 1u] = dg.y;
            acc[base + 2u] = dg.z;
            // the floor holds the transmission still: no gradient through it
            var dv = dx * gain * l;
            if (tr.vig.x <= 0.05) { dv.x = 0.0; }
            if (tr.vig.y <= 0.05) { dv.y = 0.0; }
            if (tr.vig.z <= 0.05) { dv.z = 0.0; }
            let r4 = r2 * r2;
            let r6 = r4 * r2;
            acc[base + 3u] = dv.x * r2;
            acc[base + 4u] = dv.y * r2;
            acc[base + 5u] = dv.z * r2;
            acc[base + 6u] = dv.x * r4;
            acc[base + 7u] = dv.y * r4;
            acc[base + 8u] = dv.z * r4;
            acc[base + 9u] = dv.x * r6;
            acc[base + 10u] = dv.y * r6;
            acc[base + 11u] = dv.z * r6;
            let dl = dx * gain * tr.vig;
            io[pix * 4u] = dl.x;
            io[pix * 4u + 1u] = dl.y;
            io[pix * 4u + 2u] = dl.z;
        }
    }
    workgroupBarrier();
    if (p.v.mode == 1u && t < 45u) {
        var sum = 0.0;
        for (var j = 0u; j < 64u; j = j + 1u) {
            sum = sum + acc[j * 45u + t];
        }
        partial[(wid.y * nwg.x + wid.x) * 45u + t] = sum;
    }
}

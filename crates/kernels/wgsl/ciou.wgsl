// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  CIoU loss value per assigned anchor
// @how   one thread per output element
// @opt   3
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
//
// CIoU loss value per assigned anchor:  loss = 1 - CIoU
//   CIoU = IoU - rho^2/c^2 - alpha * v
//   v     = (4/pi^2) * (atan(wg/hg) - atan(wp/hp))^2
//   alpha = v / ((1 - IoU) + v)
//   c^2   = squared diagonal of the smallest enclosing box.
// pred[A,4] = (x1,y1,x2,y2), tgt[A,4] likewise. Output out[A]. One thread/anchor.
//
// NOTE: the CPU JIT MathFunction set has NO `atan` (and supports no user
// function calls), so atan is polyfilled INLINE below: argument reduction to
// [0,1] then a degree-7 odd polynomial. |err| < 1e-3 over all positive ratios,
// accurate enough that the ciou_grad central-difference check passes < 2e-2.

struct Params {
    A: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       pred: array<f32>;
@group(0) @binding(2) var<storage, read>       tgt:  array<f32>;
@group(0) @binding(3) var<storage, read_write> out:  array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let a = gidx;
    if (a >= p.A) { return; }
    let b = a * 4u;

    let px1 = pred[b + 0u];
    let py1 = pred[b + 1u];
    let px2 = pred[b + 2u];
    let py2 = pred[b + 3u];
    let gx1 = tgt[b + 0u];
    let gy1 = tgt[b + 1u];
    let gx2 = tgt[b + 2u];
    let gy2 = tgt[b + 3u];

    let wp = px2 - px1;
    let hp = py2 - py1;
    let wg = gx2 - gx1;
    let hg = gy2 - gy1;

    // intersection
    let ix2 = min(px2, gx2);
    let ix1 = max(px1, gx1);
    let iy2 = min(py2, gy2);
    let iy1 = max(py1, gy1);
    let iw = max(0.0, ix2 - ix1);
    let ih = max(0.0, iy2 - iy1);
    let inter = iw * ih;

    let ap = wp * hp;
    let ag = wg * hg;
    let uni = max(ap + ag - inter, 1e-9);
    let iou = inter / uni;

    // center distance
    let cpx = (px1 + px2) * 0.5;
    let cpy = (py1 + py2) * 0.5;
    let cgx = (gx1 + gx2) * 0.5;
    let cgy = (gy1 + gy2) * 0.5;
    let dx = cpx - cgx;
    let dy = cpy - cgy;
    let rho2 = dx * dx + dy * dy;

    // enclosing box diagonal squared
    let ex1 = min(px1, gx1);
    let ey1 = min(py1, gy1);
    let ex2 = max(px2, gx2);
    let ey2 = max(py2, gy2);
    let cw = ex2 - ex1;
    let ch = ey2 - ey1;
    let c2 = max(cw * cw + ch * ch, 1e-9);

    // ---- atan(wg/hg) inlined (branchless; reduce arg to [0,1]) ----
    let rg = wg / max(hg, 1e-9);
    let bigg = rg > 1.0;
    let xg = select(rg, 1.0 / rg, bigg);
    let zg = xg * xg;
    var pg = 0.0028662257;
    pg = pg * zg - 0.0161657367;
    pg = pg * zg + 0.0429096138;
    pg = pg * zg - 0.0752896400;
    pg = pg * zg + 0.1065626393;
    pg = pg * zg - 0.1420889944;
    pg = pg * zg + 0.1999355085;
    pg = pg * zg - 0.3333314528;
    pg = pg * zg + 1.0;
    let polyg = pg * xg;
    let atg = select(polyg, 1.5707963267948966 - polyg, bigg);

    // ---- atan(wp/hp) inlined (branchless) ----
    let rp = wp / max(hp, 1e-9);
    let bigp = rp > 1.0;
    let xp = select(rp, 1.0 / rp, bigp);
    let zp = xp * xp;
    var pp = 0.0028662257;
    pp = pp * zp - 0.0161657367;
    pp = pp * zp + 0.0429096138;
    pp = pp * zp - 0.0752896400;
    pp = pp * zp + 0.1065626393;
    pp = pp * zp - 0.1420889944;
    pp = pp * zp + 0.1999355085;
    pp = pp * zp - 0.3333314528;
    pp = pp * zp + 1.0;
    let polyp = pp * xp;
    let atp = select(polyp, 1.5707963267948966 - polyp, bigp);

    let diff = atg - atp;
    let four_over_pi2 = 0.4052847345693511; // 4 / pi^2
    let v = four_over_pi2 * diff * diff;
    let alpha = v / max((1.0 - iou) + v, 1e-9);

    let ciou = iou - rho2 / c2 - alpha * v;
    out[a] = 1.0 - ciou;
}

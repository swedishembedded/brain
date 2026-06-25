// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Gradient of the CIoU loss L = 1 - CIoU w.r.t. the 4 predicted coords
// pred[A,4] = (x1,y1,x2,y2). Target tgt[A,4] is constant. Output dpred[A,4].
// One thread per anchor.
//
// CIoU = IoU - rho^2/c^2 - alpha*v,  L = 1 - CIoU
//   dL = -dIoU + d(rho^2/c^2) + alpha*dv      (alpha detached, standard YOLO)
//
// Per-coordinate hand-derived partials (see test for the FD gate):
//   IoU = I/U,  U = Ap + Ag - I
//     dI/dx1 = -ih*[px1>gx1]   dI/dx2 = +ih*[px2<gx2]
//     dI/dy1 = -iw*[py1>gy1]   dI/dy2 = +iw*[py2<gy2]
//     dAp/dx1 = -hp  dAp/dx2 = +hp  dAp/dy1 = -wp  dAp/dy2 = +wp
//     dIoU = (dI*U - I*dU)/U^2,  dU = dAp - dI
//   rho^2 = dcx^2 + dcy^2,  dcx=(x1+x2)/2 - cgx
//     drho2/dx1 = drho2/dx2 = dcx ; drho2/dy1 = drho2/dy2 = dcy
//   c^2 = cw^2 + ch^2 (enclosing box)
//     dc2/dx1 = -2cw*[px1<gx1]  dc2/dx2 = +2cw*[px2>gx2]
//     dc2/dy1 = -2ch*[py1<gy1]  dc2/dy2 = +2ch*[py2>gy2]
//   d(rho2/c2) = (drho2*c2 - rho2*dc2)/c2^2
//   v = k*(atg-atp)^2, k=4/pi^2, atp=atan(u), u=wp/hp
//     du/dwp=1/hp, du/dhp=-wp/hp^2, datan(u)=du/(1+u^2)
//     dv = 2k(atg-atp)*(-datp)

struct Params {
    A: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       pred:  array<f32>;
@group(0) @binding(2) var<storage, read>       tgt:   array<f32>;
@group(0) @binding(3) var<storage, read_write> dpred: array<f32>;

// NOTE: the CPU JIT supports no user function calls, so the atan polyfill is
// inlined below. Its analytic derivative 1/(1+u^2) is EXACT regardless of the
// polynomial value approximation, so the gradient is consistent with the value
// kernel's polyfill, which keeps the FD check tight.

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

    // intersection / union / IoU
    let ix1 = max(px1, gx1);
    let iy1 = max(py1, gy1);
    let ix2 = min(px2, gx2);
    let iy2 = min(py2, gy2);
    let iw = max(0.0, ix2 - ix1);
    let ih = max(0.0, iy2 - iy1);
    let inter = iw * ih;
    let ap = wp * hp;
    let ag = wg * hg;
    let uni = max(ap + ag - inter, 1e-9);
    let iou = inter / uni;

    // active-edge indicators (1.0/0.0)
    let m_ix1 = select(0.0, 1.0, px1 > gx1);   // px1 is the inner-left
    let m_ix2 = select(0.0, 1.0, px2 < gx2);
    let m_iy1 = select(0.0, 1.0, py1 > gy1);
    let m_iy2 = select(0.0, 1.0, py2 < gy2);
    let pos = select(0.0, 1.0, (iw > 0.0) && (ih > 0.0));

    // dI/d coord
    let dI_x1 = -ih * m_ix1 * pos;
    let dI_x2 =  ih * m_ix2 * pos;
    let dI_y1 = -iw * m_iy1 * pos;
    let dI_y2 =  iw * m_iy2 * pos;
    // dAp/d coord
    let dAp_x1 = -hp;
    let dAp_x2 =  hp;
    let dAp_y1 = -wp;
    let dAp_y2 =  wp;
    // dU = dAp - dI
    let dU_x1 = dAp_x1 - dI_x1;
    let dU_x2 = dAp_x2 - dI_x2;
    let dU_y1 = dAp_y1 - dI_y1;
    let dU_y2 = dAp_y2 - dI_y2;
    // dIoU = (dI*U - I*dU)/U^2
    let u2 = uni * uni;
    let dIoU_x1 = (dI_x1 * uni - inter * dU_x1) / u2;
    let dIoU_x2 = (dI_x2 * uni - inter * dU_x2) / u2;
    let dIoU_y1 = (dI_y1 * uni - inter * dU_y1) / u2;
    let dIoU_y2 = (dI_y2 * uni - inter * dU_y2) / u2;

    // center distance
    let cpx = (px1 + px2) * 0.5;
    let cpy = (py1 + py2) * 0.5;
    let cgx = (gx1 + gx2) * 0.5;
    let cgy = (gy1 + gy2) * 0.5;
    let dcx = cpx - cgx;
    let dcy = cpy - cgy;
    let rho2 = dcx * dcx + dcy * dcy;
    // drho2
    let drho2_x1 = dcx;
    let drho2_x2 = dcx;
    let drho2_y1 = dcy;
    let drho2_y2 = dcy;

    // enclosing box
    let ex1 = min(px1, gx1);
    let ey1 = min(py1, gy1);
    let ex2 = max(px2, gx2);
    let ey2 = max(py2, gy2);
    let cw = ex2 - ex1;
    let ch = ey2 - ey1;
    let c2 = max(cw * cw + ch * ch, 1e-9);
    let m_ex1 = select(0.0, 1.0, px1 < gx1);
    let m_ex2 = select(0.0, 1.0, px2 > gx2);
    let m_ey1 = select(0.0, 1.0, py1 < gy1);
    let m_ey2 = select(0.0, 1.0, py2 > gy2);
    let dc2_x1 = -2.0 * cw * m_ex1;
    let dc2_x2 =  2.0 * cw * m_ex2;
    let dc2_y1 = -2.0 * ch * m_ey1;
    let dc2_y2 =  2.0 * ch * m_ey2;
    // d(rho2/c2) = (drho2*c2 - rho2*dc2)/c2^2
    let c2sq = c2 * c2;
    let dR_x1 = (drho2_x1 * c2 - rho2 * dc2_x1) / c2sq;
    let dR_x2 = (drho2_x2 * c2 - rho2 * dc2_x2) / c2sq;
    let dR_y1 = (drho2_y1 * c2 - rho2 * dc2_y1) / c2sq;
    let dR_y2 = (drho2_y2 * c2 - rho2 * dc2_y2) / c2sq;

    // aspect ratio term
    let hp_s = max(hp, 1e-9);
    let u = wp / hp_s;

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

    // ---- atan(u = wp/hp) inlined (branchless) ----
    let rp = u;
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

    let four_over_pi2 = 0.4052847345693511;
    let v = four_over_pi2 * (atg - atp) * (atg - atp);
    let alpha = v / max((1.0 - iou) + v, 1e-9);
    // dv = 2k(atg-atp)*(-datp),  datp = (du)/(1+u^2)
    let inv1u2 = 1.0 / (1.0 + u * u);
    let coef = 2.0 * four_over_pi2 * (atg - atp) * (-1.0) * inv1u2;
    // du/dx via wp,hp: dwp/dx1=-1,dwp/dx2=1 ; dhp/dy1=-1,dhp/dy2=1
    let du_x1 = -1.0 / hp_s;
    let du_x2 =  1.0 / hp_s;
    let du_y1 =  wp / (hp_s * hp_s);   // -wp/hp^2 * (dhp/dy1=-1)
    let du_y2 = -wp / (hp_s * hp_s);
    let dv_x1 = coef * du_x1;
    let dv_x2 = coef * du_x2;
    let dv_y1 = coef * du_y1;
    let dv_y2 = coef * du_y2;

    // dL = -dIoU + d(rho2/c2) + alpha*dv
    dpred[b + 0u] = -dIoU_x1 + dR_x1 + alpha * dv_x1;
    dpred[b + 1u] = -dIoU_y1 + dR_y1 + alpha * dv_y1;
    dpred[b + 2u] = -dIoU_x2 + dR_x2 + alpha * dv_x2;
    dpred[b + 3u] = -dIoU_y2 + dR_y2 + alpha * dv_y2;
}

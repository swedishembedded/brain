// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Pure-Rust scalar rasterizer — the test oracle. Mirrors `splat_project.wgsl`
//! + `splat_naive.wgsl` line for line (same culls, same thresholds, same
//! termination semantics) so device kernels can be diffed against it exactly.

use crate::types::{Camera, Mode, RenderOpts, Splats};

/// Projected gaussian record, identical to the device `proj` stride-9 layout.
#[derive(Clone, Copy, Debug, Default)]
pub struct Proj {
    pub x: f32,
    pub y: f32,
    pub conic: [f32; 3],
    pub opacity: f32,
    pub depth: f32,
    pub radius: [f32; 2],
}

/// Project gaussian `i`; `None` when culled (any reason).
pub fn project_one(s: &Splats, i: usize, cam: &Camera, o: &RenderOpts) -> Option<Proj> {
    let v = cam.viewmat();
    let m = &s.means[i * 3..i * 3 + 3];
    let x = v[0] * m[0] + v[1] * m[1] + v[2] * m[2] + v[3];
    let y = v[4] * m[0] + v[5] * m[1] + v[6] * m[2] + v[7];
    let z = v[8] * m[0] + v[9] * m[1] + v[10] * m[2] + v[11];
    if z < o.near || z > o.far {
        return None;
    }

    let q = &s.quats[i * 4..i * 4 + 4];
    let qn = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt() + 1e-8;
    let (qw, qx, qy, qz) = (q[0] / qn, q[1] / qn, q[2] / qn, q[3] / qn);
    let rq = [
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
        [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
        [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
    ];
    let sc = &s.scales[i * 3..i * 3 + 3];
    let s2 = [sc[0] * sc[0], sc[1] * sc[1], sc[2] * sc[2]];
    let mut sig3 = [[0.0f32; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            sig3[a][b] = (0..3).map(|k| s2[k] * rq[a][k] * rq[b][k]).sum();
        }
    }
    let r = [[v[0], v[1], v[2]], [v[4], v[5], v[6]], [v[8], v[9], v[10]]];
    let mut a3 = [[0.0f32; 3]; 3];
    for i3 in 0..3 {
        for j3 in 0..3 {
            a3[i3][j3] = (0..3).map(|k| r[i3][k] * sig3[k][j3]).sum();
        }
    }
    let mut cc = [[0.0f32; 3]; 3];
    for i3 in 0..3 {
        for j3 in 0..3 {
            cc[i3][j3] = (0..3).map(|k| a3[i3][k] * r[j3][k]).sum();
        }
    }

    let rz = 1.0 / z;
    let (w, h) = (cam.width as f32, cam.height as f32);
    let tan_fovx = 0.5 * w / cam.fx;
    let tan_fovy = 0.5 * h / cam.fy;
    let lim_x_pos = (w - cam.cx) / cam.fx + 0.3 * tan_fovx;
    let lim_x_neg = cam.cx / cam.fx + 0.3 * tan_fovx;
    let lim_y_pos = (h - cam.cy) / cam.fy + 0.3 * tan_fovy;
    let lim_y_neg = cam.cy / cam.fy + 0.3 * tan_fovy;
    let txc = z * (x * rz).clamp(-lim_x_neg, lim_x_pos);
    let tyc = z * (y * rz).clamp(-lim_y_neg, lim_y_pos);
    let j00 = cam.fx * rz;
    let j02 = -cam.fx * txc * rz * rz;
    let j11 = cam.fy * rz;
    let j12 = -cam.fy * tyc * rz * rz;
    let mut sa = j00 * j00 * cc[0][0] + 2.0 * j00 * j02 * cc[0][2] + j02 * j02 * cc[2][2];
    let sb = j00 * (cc[0][1] * j11 + cc[0][2] * j12) + j02 * (cc[2][1] * j11 + cc[2][2] * j12);
    let mut scv = j11 * j11 * cc[1][1] + 2.0 * j11 * j12 * cc[1][2] + j12 * j12 * cc[2][2];

    let det_orig = sa * scv - sb * sb;
    sa += o.eps2d;
    scv += o.eps2d;
    let det_blur = sa * scv - sb * sb;
    if det_blur <= 0.0 {
        return None;
    }
    let comp = (det_orig / det_blur).max(0.005 * 0.005).sqrt();

    let mut op = s.opacities[i];
    if o.antialiased {
        op *= comp;
    }
    if op < 1.0 / 255.0 {
        return None;
    }
    let extend = (2.0 * (op * 255.0).ln()).max(0.0).sqrt().min(3.33);
    let rx = (extend * sa.sqrt()).ceil();
    let ry = (extend * scv.sqrt()).ceil();
    if rx <= 0.0 || ry <= 0.0 {
        return None;
    }
    let px = cam.fx * x * rz + cam.cx;
    let py = cam.fy * y * rz + cam.cy;
    if px + rx <= 0.0 || px - rx >= w || py + ry <= 0.0 || py - ry >= h {
        return None;
    }
    Some(Proj {
        x: px,
        y: py,
        conic: [scv / det_blur, -sb / det_blur, sa / det_blur],
        opacity: op,
        depth: z,
        radius: [rx, ry],
    })
}

/// Render the full scene front-to-back (sorts by depth internally).
/// Returns RGBA f32, `w*h*4`.
pub fn render(s: &Splats, cam: &Camera, o: &RenderOpts) -> Vec<f32> {
    let mut projected: Vec<(usize, Proj)> = (0..s.len())
        .filter_map(|i| project_one(s, i, cam, o).map(|p| (i, p)))
        .collect();
    projected.sort_by(|a, b| a.1.depth.total_cmp(&b.1.depth));

    let (w, h) = (cam.width as usize, cam.height as usize);
    let mut img = vec![0.0f32; w * h * 4];
    for py in 0..h {
        for px in 0..w {
            let (fx, fy) = (px as f32 + 0.5, py as f32 + 0.5);
            let mut t = 1.0f32;
            let mut c = [0.0f32; 3];
            let mut dep = 0.0f32;
            for (gi, p) in &projected {
                let dx = p.x - fx;
                let dy = p.y - fy;
                if dx.abs() > p.radius[0] || dy.abs() > p.radius[1] {
                    continue;
                }
                let sigma = 0.5 * (p.conic[0] * dx * dx + p.conic[2] * dy * dy) + p.conic[1] * dx * dy;
                if sigma < 0.0 {
                    continue;
                }
                let alpha = (p.opacity * (-sigma).exp()).min(0.99);
                if alpha < 1.0 / 255.0 {
                    continue;
                }
                let next_t = t * (1.0 - alpha);
                if next_t <= 1e-4 {
                    break;
                }
                let wgt = alpha * t;
                for k in 0..3 {
                    c[k] += s.colors[gi * 3 + k] * wgt;
                }
                dep += p.depth * wgt;
                t = next_t;
            }
            let a = 1.0 - t;
            let px4 = (py * w + px) * 4;
            match o.mode {
                Mode::Color => {
                    for k in 0..3 {
                        img[px4 + k] = c[k] + t * o.bg[k];
                    }
                }
                Mode::Depth => {
                    let d = if a > 1e-6 { dep / a } else { 0.0 };
                    img[px4] = d;
                    img[px4 + 1] = d;
                    img[px4 + 2] = d;
                }
            }
            img[px4 + 3] = a;
        }
    }
    img
}

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Pure-Rust scalar rasterizer - the test oracle. Mirrors
//! `splat_project.wgsl` + `splat_naive.wgsl` line for line (same culls, same
//! thresholds, same termination semantics) so device kernels can be diffed
//! against it exactly.

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
                for (k, ck) in c.iter_mut().enumerate() {
                    *ck += s.colors[gi * 3 + k] * wgt;
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

/// One gaussian as the ray renderer sees it from a camera, in f64: its mean
/// in the camera frame, the precision of its filtered covariance, the
/// compensated opacity and its camera-facing normal.
#[derive(Clone, Copy, Debug)]
pub struct RayGaussian {
    pub index: usize,
    pub m: [f64; 3],
    pub a: [[f64; 3]; 3],
    pub opacity: f64,
    pub normal: [f64; 3],
    pub range: f64,
}

/// [`RayGaussian`] of gaussian `i` under `filter3d` (a variance, 0 = none),
/// or `None` where `splat_ray_project.wgsl` culls it for depth or opacity.
/// Image-space bounds are not modelled: the oracle evaluates every gaussian
/// at every pixel, so a bound that clips something visible shows up as a
/// difference instead of being reproduced.
pub fn ray_gaussian(s: &Splats, i: usize, filter3d: f64, cam: &Camera, o: &RenderOpts) -> Option<RayGaussian> {
    let v = cam.viewmat().map(|x| x as f64);
    let mw = [s.means[i * 3] as f64, s.means[i * 3 + 1] as f64, s.means[i * 3 + 2] as f64];
    let m: [f64; 3] = std::array::from_fn(|r| v[r * 4] * mw[0] + v[r * 4 + 1] * mw[1] + v[r * 4 + 2] * mw[2] + v[r * 4 + 3]);
    let range = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
    if cam.lens.is_perspective() {
        if m[2] < o.near as f64 || m[2] > o.far as f64 {
            return None;
        }
    } else if range < o.near as f64 || range > o.far as f64 {
        return None;
    }
    let q: [f64; 4] = std::array::from_fn(|k| s.quats[i * 4 + k] as f64);
    let qn = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt() + 1e-8;
    let (w, x, y, z) = (q[0] / qn, q[1] / qn, q[2] / qn, q[3] / qn);
    let cols = [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)],
        [2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)],
        [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)],
    ];
    let ax: Vec<[f64; 3]> = cols
        .iter()
        .map(|c| std::array::from_fn(|r| v[r * 4] * c[0] + v[r * 4 + 1] * c[1] + v[r * 4 + 2] * c[2]))
        .collect();
    let sv: [f64; 3] = std::array::from_fn(|k| s.scales[i * 3 + k] as f64);
    let lf: [f64; 3] = std::array::from_fn(|k| sv[k] * sv[k] + filter3d);
    let c3 = if filter3d > 0.0 { ((0..3).map(|k| sv[k] * sv[k] / lf[k]).product::<f64>()).sqrt() } else { 1.0 };
    let ppr = cam.intrinsics().pixels_per_radian(m)?;
    let fp = o.eps2d as f64 * (range / ppr).powi(2);
    let mut op = s.opacities[i] as f64 * c3;
    if o.antialiased {
        let nh = [m[0] / range, m[1] / range, m[2] / range];
        let c: Vec<f64> = ax.iter().map(|a| a[0] * nh[0] + a[1] * nh[1] + a[2] * nh[2]).collect();
        let det: f64 = lf.iter().product();
        let dd = det * (0..3).map(|k| c[k] * c[k] / lf[k]).sum::<f64>();
        let tt = lf.iter().sum::<f64>() - (0..3).map(|k| c[k] * c[k] * lf[k]).sum::<f64>();
        op *= (dd / (dd + fp * tt + fp * fp)).sqrt();
    }
    if op < 1.0 / 255.0 {
        return None;
    }
    let mut a = [[0.0f64; 3]; 3];
    for k in 0..3 {
        let wk = 1.0 / (lf[k] + fp);
        for r in 0..3 {
            for c in 0..3 {
                a[r][c] += wk * ax[k][r] * ax[k][c];
            }
        }
    }
    let kmin = if sv[0] <= sv[1] && sv[0] <= sv[2] { 0 } else if sv[1] <= sv[2] { 1 } else { 2 };
    let mut normal = ax[kmin];
    if normal[0] * m[0] + normal[1] * m[1] + normal[2] * m[2] > 0.0 {
        normal = normal.map(|v| -v);
    }
    Some(RayGaussian { index: i, m, a, opacity: op, normal, range })
}

/// The ray pixel `(px, py)` casts in the mid-frame camera: origin and unit
/// direction, `None` where the lens has no ray.
pub fn pixel_ray(cam: &Camera, px: f64, py: f64) -> Option<([f64; 3], [f64; 3])> {
    let d = cam.intrinsics().unproject([px, py])?;
    let tau = py / cam.height as f64 - 0.5;
    let w: [f64; 3] = std::array::from_fn(|k| tau * cam.shutter[k] as f64);
    let o: [f64; 3] = std::array::from_fn(|k| tau * cam.shutter[3 + k] as f64);
    Some((o, rotate(w, d)))
}

fn rotate(w: [f64; 3], x: [f64; 3]) -> [f64; 3] {
    let th = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
    if th < 1e-12 {
        return x;
    }
    let k = [w[0] / th, w[1] / th, w[2] / th];
    let (c, s) = (th.cos(), th.sin());
    let kx = [k[1] * x[2] - k[2] * x[1], k[2] * x[0] - k[0] * x[2], k[0] * x[1] - k[1] * x[0]];
    let kd = k[0] * x[0] + k[1] * x[1] + k[2] * x[2];
    std::array::from_fn(|r| x[r] * c + kx[r] * s + k[r] * kd * (1.0 - c))
}

/// The order a pixel of [`render_ray_ordered`] composites its gaussians in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RayOrder {
    /// By the range `t*` along the pixel's own ray at which each gaussian
    /// responds most: the order light meets them, and the one the 2DGS
    /// distortion assumes. It differs from pixel to pixel wherever gaussians
    /// overlap or intersect.
    PerPixel,
    /// By range to each gaussian's mean, one order for the whole frame: what a
    /// global sort of the centres alone composites. Where two gaussians
    /// overlap it is wrong on part of the overlap, and as the view moves it
    /// flips for all of the overlap at once.
    Mean,
}

/// The ray renderer, in f64 and without tiles: `(rgba [W*H*4], aux
/// [W*H*5])` exactly as `splat_ray_rasterize.wgsl` defines them, every pixel
/// compositing in its own exact range order ([`RayOrder::PerPixel`]).
pub fn render_ray(s: &Splats, filter3d: &[f32], cam: &Camera, o: &RenderOpts) -> (Vec<f32>, Vec<f32>) {
    render_ray_ordered(s, filter3d, cam, o, RayOrder::PerPixel)
}

/// One pair a pixel composites: its range `t*`, its compositing weight
/// `w = alpha T` and its gaussian.
#[derive(Clone, Copy, Debug)]
pub struct RayHit<'a> {
    pub t: f64,
    pub w: f64,
    pub g: &'a RayGaussian,
}

/// The gaussians of `s` the ray renderer evaluates from `cam`, in the order
/// of their means' range.
pub fn ray_gaussians(s: &Splats, filter3d: &[f32], cam: &Camera, o: &RenderOpts) -> Vec<RayGaussian> {
    let mut gs: Vec<RayGaussian> = (0..s.len())
        .filter_map(|i| ray_gaussian(s, i, filter3d.get(i).copied().unwrap_or(0.0) as f64, cam, o))
        .collect();
    gs.sort_by(|a, b| (a.range as f32).total_cmp(&(b.range as f32)).then(a.index.cmp(&b.index)));
    gs
}

/// What pixel `(px, py)` composites, in `order`, until its walk stops: every
/// pair's range, weight and gaussian, and the transmittance left. `None` =
/// the lens has no ray there.
pub fn ray_pixel<'a>(gs: &'a [RayGaussian], cam: &Camera, px: usize, py: usize, order: RayOrder) -> (Vec<RayHit<'a>>, f64) {
    let mut hits: Vec<(f64, f64, &RayGaussian)> = Vec::new();
    if let Some((org, d)) = pixel_ray(cam, px as f64 + 0.5, py as f64 + 0.5) {
        for g in gs {
            let dm: [f64; 3] = std::array::from_fn(|k| g.m[k] - org[k]);
            let ad: [f64; 3] = std::array::from_fn(|r| (0..3).map(|k| g.a[r][k] * d[k]).sum());
            let vv: f64 = (0..3).map(|k| d[k] * ad[k]).sum();
            if vv <= 0.0 {
                continue;
            }
            let tt = (0..3).map(|k| dm[k] * ad[k]).sum::<f64>() / vv;
            if tt <= 0.0 {
                continue;
            }
            let e: [f64; 3] = std::array::from_fn(|k| dm[k] - tt * d[k]);
            let q: f64 = (0..3).map(|r| (0..3).map(|k| e[r] * g.a[r][k] * e[k]).sum::<f64>()).sum();
            let alpha = (g.opacity * (-0.5 * q).exp()).min(0.99);
            if alpha >= 1.0 / 255.0 {
                hits.push((tt, alpha, g));
            }
        }
    }
    if order == RayOrder::PerPixel {
        // stable: equal ranges keep the order of the means
        hits.sort_by(|a, b| a.0.total_cmp(&b.0));
    }
    let mut t = 1.0f64;
    let mut out = Vec::with_capacity(hits.len());
    for (tt, alpha, g) in hits {
        let next = t * (1.0 - alpha);
        if next <= 1e-4 {
            break;
        }
        out.push(RayHit { t: tt, w: alpha * t, g });
        t = next;
    }
    (out, t)
}

/// [`render_ray`] compositing in `order`.
pub fn render_ray_ordered(s: &Splats, filter3d: &[f32], cam: &Camera, o: &RenderOpts, order: RayOrder) -> (Vec<f32>, Vec<f32>) {
    let gs = ray_gaussians(s, filter3d, cam, o);
    let (w, h) = (cam.width as usize, cam.height as usize);
    let mut img = vec![0.0f32; w * h * 4];
    let mut aux = vec![0.0f32; w * h * 5];
    for py in 0..h {
        for px in 0..w {
            let pix = py * w + px;
            let (hits, t) = ray_pixel(&gs, cam, px, py, order);
            let (mut c, mut nsum) = ([0.0f64; 3], [0.0f64; 3]);
            let (mut dsum, mut wsum, mut dist) = (0.0f64, 0.0f64, 0.0f64);
            for hit in &hits {
                for k in 0..3 {
                    c[k] += hit.w * s.colors[hit.g.index * 3 + k] as f64;
                    nsum[k] += hit.w * hit.g.normal[k];
                }
                dist += 2.0 * hit.w * (hit.t * wsum - dsum);
                dsum += hit.w * hit.t;
                wsum += hit.w;
            }
            for k in 0..3 {
                img[pix * 4 + k] = (c[k] + t * o.bg[k] as f64) as f32;
            }
            let a = 1.0 - t;
            img[pix * 4 + 3] = a as f32;
            aux[pix * 5] = if a > 1e-6 { (dsum / a) as f32 } else { 0.0 };
            for k in 0..3 {
                aux[pix * 5 + 1 + k] = nsum[k] as f32;
            }
            aux[pix * 5 + 4] = dist as f32;
        }
    }
    (img, aux)
}

/// `splat_ray_diagnose.wgsl` in f64, every pixel in its exact range order:
/// `[W*H*12]` as `Renderer::diagnose` documents it.
pub fn diagnose_ray(s: &Splats, filter3d: &[f32], cam: &Camera, o: &RenderOpts) -> Vec<f32> {
    let gs = ray_gaussians(s, filter3d, cam, o);
    let (w, h) = (cam.width as usize, cam.height as usize);
    let mut out = vec![0.0f32; w * h * 12];
    for py in 0..h {
        for px in 0..w {
            let (hits, t) = ray_pixel(&gs, cam, px, py, RayOrder::PerPixel);
            let d = &mut out[(py * w + px) * 12..(py * w + px + 1) * 12];
            let sw: f64 = hits.iter().map(|h| h.w).sum();
            d[0] = (1.0 - t) as f32;
            let mut acc = 0.0f64;
            for h in &hits {
                acc += h.w;
                if acc >= 0.5 {
                    d[2] = h.t as f32;
                    break;
                }
            }
            d[5] = hits.len() as f32;
            d[7] = f32::from_bits(u32::MAX);
            if sw > 1e-8 {
                let mean = hits.iter().map(|h| h.w * h.t).sum::<f64>() / sw;
                let var = hits.iter().map(|h| h.w * h.t * h.t).sum::<f64>() / sw - mean * mean;
                let ent = -hits.iter().map(|h| (h.w / sw) * (h.w / sw).ln()).sum::<f64>();
                let top = hits.iter().max_by(|a, b| a.w.total_cmp(&b.w)).expect("a weight");
                let n: [f64; 3] = std::array::from_fn(|k| hits.iter().map(|h| h.w * h.g.normal[k]).sum());
                let nl = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                d[1] = mean as f32;
                d[3] = var.max(0.0).sqrt() as f32;
                d[4] = ent.max(0.0) as f32;
                d[6] = (top.w / sw) as f32;
                d[7] = f32::from_bits(top.g.index as u32);
                if nl > 1e-8 {
                    for k in 0..3 {
                        d[8 + k] = (n[k] / nl) as f32;
                    }
                }
            }
            d[11] = sw as f32;
        }
    }
    out
}

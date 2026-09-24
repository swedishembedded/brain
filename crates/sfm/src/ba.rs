// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Bundle adjustment: Levenberg-Marquardt on the pixel reprojection error of
//! every observation, over every camera pose, every point and (optionally)
//! the shared intrinsics `f, k1, k2` - Triggs et al., "Bundle Adjustment - A
//! Modern Synthesis" (1999), §4 and §6.
//!
//! The points are eliminated with the Schur complement, so the system solved
//! is the dense reduced CAMERA system (6 per pose + 3 intrinsics); each point
//! is then recovered from its own 3x3 block. Residuals pass through a Huber
//! loss (IRLS weights) so a mismatch that survived verification pulls with
//! bounded force. The first pose in `fixed` pins the gauge's rotation and
//! translation; scale is held by the damping.

use crate::camera::{Intrinsics, Pose};
use crate::linalg::{cholesky_solve, exp_so3, inv3_spd, mm, M3, V3};

/// One image measurement of one point.
#[derive(Clone, Copy, Debug)]
pub struct Observation {
    pub cam: usize,
    pub point: usize,
    pub px: [f64; 2],
}

#[derive(Clone, Debug)]
pub struct BaCfg {
    pub iters: usize,
    /// Huber threshold in pixels.
    pub huber: f64,
    /// Refine `f`, `k1`, `k2`.
    pub intrinsics: bool,
    /// Poses held fixed (gauge and already-trusted cameras).
    pub fixed: Vec<usize>,
}

impl Default for BaCfg {
    fn default() -> Self {
        BaCfg { iters: 50, huber: 2.0, intrinsics: true, fixed: vec![0] }
    }
}

/// Reprojection statistics before and after.
#[derive(Clone, Copy, Debug)]
pub struct BaReport {
    pub rms_before: f64,
    pub rms_after: f64,
    pub iterations: usize,
}

/// Residual and Jacobians of one observation.
struct Lin {
    r: [f64; 2],
    /// d r / d (omega, t)
    jc: [[f64; 6]; 2],
    /// d r / d (f, k1, k2)
    jk: [[f64; 3]; 2],
    /// d r / d X
    jp: [[f64; 3]; 2],
}

fn linearize(k: &Intrinsics, p: &Pose, x: V3, obs: [f64; 2]) -> Option<Lin> {
    let c = p.to_cam(x);
    if c[2] <= 1e-9 {
        return None;
    }
    let iz = 1.0 / c[2];
    let (nx, ny) = (c[0] * iz, c[1] * iz);
    let r2 = nx * nx + ny * ny;
    let d = 1.0 + k.k1 * r2 + k.k2 * r2 * r2;
    let dd = k.k1 + 2.0 * k.k2 * r2;
    let u = k.f * d * nx + k.cx;
    let v = k.f * d * ny + k.cy;
    // d(u,v)/d(nx,ny)
    let a = [[k.f * (d + 2.0 * nx * nx * dd), k.f * 2.0 * nx * ny * dd], [k.f * 2.0 * nx * ny * dd, k.f * (d + 2.0 * ny * ny * dd)]];
    // d(nx,ny)/dc
    let b = [[iz, 0.0, -nx * iz], [0.0, iz, -ny * iz]];
    let mut jcam = [[0.0f64; 3]; 2];
    for i in 0..2 {
        for j in 0..3 {
            jcam[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
        }
    }
    let rx = [c[0] - p.t[0], c[1] - p.t[1], c[2] - p.t[2]];
    // dc/domega = -[R X]x
    let dw = [[0.0, rx[2], -rx[1]], [-rx[2], 0.0, rx[0]], [rx[1], -rx[0], 0.0]];
    let mut jc = [[0.0f64; 6]; 2];
    let mut jp = [[0.0f64; 3]; 2];
    for i in 0..2 {
        for j in 0..3 {
            jc[i][j] = (0..3).map(|m| jcam[i][m] * dw[m][j]).sum();
            jc[i][3 + j] = jcam[i][j];
            jp[i][j] = (0..3).map(|m| jcam[i][m] * p.r[m * 3 + j]).sum();
        }
    }
    let jk = [[d * nx, k.f * nx * r2, k.f * nx * r2 * r2], [d * ny, k.f * ny * r2, k.f * ny * r2 * r2]];
    Some(Lin { r: [u - obs[0], v - obs[1]], jc, jk, jp })
}

fn huber_weight(r: &[f64; 2], delta: f64) -> f64 {
    let n = (r[0] * r[0] + r[1] * r[1]).sqrt();
    if n <= delta { 1.0 } else { delta / n }
}

fn huber_cost(r: &[f64; 2], delta: f64) -> f64 {
    let n2 = r[0] * r[0] + r[1] * r[1];
    let n = n2.sqrt();
    if n <= delta { n2 } else { 2.0 * delta * n - delta * delta }
}

fn total_cost(k: &Intrinsics, poses: &[Pose], pts: &[V3], obs: &[Observation], delta: f64) -> f64 {
    obs.iter()
        .map(|o| match crate::camera::project(k, &poses[o.cam], pts[o.point]) {
            Some(uv) => huber_cost(&[uv[0] - o.px[0], uv[1] - o.px[1]], delta),
            None => 4.0 * delta * delta * 100.0,
        })
        .sum()
}

/// RMS pixel reprojection error over `obs` (points behind a camera skipped).
pub fn rms(k: &Intrinsics, poses: &[Pose], pts: &[V3], obs: &[Observation]) -> f64 {
    let (mut s, mut n) = (0.0f64, 0usize);
    for o in obs {
        if let Some(uv) = crate::camera::project(k, &poses[o.cam], pts[o.point]) {
            s += (uv[0] - o.px[0]).powi(2) + (uv[1] - o.px[1]).powi(2);
            n += 1;
        }
    }
    (s / n.max(1) as f64).sqrt()
}

/// Refine `k`, `poses` and `points` in place against `obs`.
pub fn bundle_adjust(k: &mut Intrinsics, poses: &mut [Pose], points: &mut [V3], obs: &[Observation], cfg: &BaCfg) -> BaReport {
    let nc = poses.len();
    let mut slot = vec![None; nc];
    let mut c = 0usize;
    for (i, s) in slot.iter_mut().enumerate() {
        if !cfg.fixed.contains(&i) {
            *s = Some(c);
            c += 6;
        }
    }
    let kofs = c;
    if cfg.intrinsics {
        c += 3;
    }
    let mut by_point: Vec<Vec<usize>> = vec![Vec::new(); points.len()];
    for (i, o) in obs.iter().enumerate() {
        by_point[o.point].push(i);
    }
    let rms_before = rms(k, poses, points, obs);
    let mut lambda = 1e-3;
    let mut cost = total_cost(k, poses, points, obs, cfg.huber);
    let mut it = 0;
    while it < cfg.iters {
        it += 1;
        // ---- normal equations ----
        let mut u = vec![0.0f64; c * c];
        let mut gc = vec![0.0f64; c];
        let mut vblocks = vec![[0.0f64; 9]; points.len()];
        let mut gp = vec![[0.0f64; 3]; points.len()];
        // per observation: local camera-side columns and W = Jcᵀ Jp (cols x 3)
        let mut local: Vec<(Vec<usize>, Vec<[f64; 3]>)> = Vec::with_capacity(obs.len());
        for o in obs {
            let Some(l) = linearize(k, &poses[o.cam], points[o.point], o.px) else {
                local.push((Vec::new(), Vec::new()));
                continue;
            };
            let w = huber_weight(&l.r, cfg.huber);
            let mut cols = Vec::with_capacity(9);
            let mut jrows: Vec<[f64; 2]> = Vec::with_capacity(9);
            if let Some(s) = slot[o.cam] {
                for j in 0..6 {
                    cols.push(s + j);
                    jrows.push([l.jc[0][j], l.jc[1][j]]);
                }
            }
            if cfg.intrinsics {
                for j in 0..3 {
                    cols.push(kofs + j);
                    jrows.push([l.jk[0][j], l.jk[1][j]]);
                }
            }
            for (a, ja) in cols.iter().zip(&jrows) {
                gc[*a] += w * (ja[0] * l.r[0] + ja[1] * l.r[1]);
                for (b, jb) in cols.iter().zip(&jrows) {
                    u[a * c + b] += w * (ja[0] * jb[0] + ja[1] * jb[1]);
                }
            }
            let vb = &mut vblocks[o.point];
            for a in 0..3 {
                gp[o.point][a] += w * (l.jp[0][a] * l.r[0] + l.jp[1][a] * l.r[1]);
                for b in 0..3 {
                    vb[a * 3 + b] += w * (l.jp[0][a] * l.jp[0][b] + l.jp[1][a] * l.jp[1][b]);
                }
            }
            let wm: Vec<[f64; 3]> = jrows
                .iter()
                .map(|ja| std::array::from_fn(|b| w * (ja[0] * l.jp[0][b] + ja[1] * l.jp[1][b])))
                .collect();
            local.push((cols, wm));
        }
        // ---- damped Schur complement, retried with more damping on failure
        let mut accepted = false;
        for _attempt in 0..8 {
            let mut s = u.clone();
            for a in 0..c {
                s[a * c + a] += lambda * u[a * c + a].max(1e-9);
            }
            let mut rhs: Vec<f64> = gc.iter().map(|v| -v).collect();
            let mut vinv = vec![[0.0f64; 9]; points.len()];
            for (p, list) in by_point.iter().enumerate() {
                let mut v = vblocks[p];
                for a in 0..3 {
                    v[a * 3 + a] += lambda * v[a * 3 + a].max(1e-9);
                }
                let Some(vi) = inv3_spd(&v) else { continue };
                vinv[p] = vi;
                let bp = [-gp[p][0], -gp[p][1], -gp[p][2]];
                let vbp = mv3(&vi, bp);
                for &oa in list {
                    let (ca, wa) = &local[oa];
                    for (ia, wr) in ca.iter().zip(wa) {
                        rhs[*ia] -= wr[0] * vbp[0] + wr[1] * vbp[1] + wr[2] * vbp[2];
                    }
                    for &ob in list {
                        let (cb, wb) = &local[ob];
                        for (ia, wra) in ca.iter().zip(wa) {
                            let t = mv3(&vi, *wra);
                            for (ib, wrb) in cb.iter().zip(wb) {
                                s[ia * c + ib] -= t[0] * wrb[0] + t[1] * wrb[1] + t[2] * wrb[2];
                            }
                        }
                    }
                }
            }
            let dc = if c > 0 { cholesky_solve(&s, &rhs, c) } else { Some(Vec::new()) };
            let Some(dc) = dc else {
                lambda *= 10.0;
                continue;
            };
            // back-substitute the points
            let mut np = points.to_vec();
            for (p, list) in by_point.iter().enumerate() {
                let mut b = [-gp[p][0], -gp[p][1], -gp[p][2]];
                for &oa in list {
                    let (ca, wa) = &local[oa];
                    for (ia, wr) in ca.iter().zip(wa) {
                        for q in 0..3 {
                            b[q] -= wr[q] * dc[*ia];
                        }
                    }
                }
                let d = mv3(&vinv[p], b);
                np[p] = [points[p][0] + d[0], points[p][1] + d[1], points[p][2] + d[2]];
            }
            let mut npose = poses.to_vec();
            for (i, s) in slot.iter().enumerate() {
                if let Some(s) = *s {
                    let r: M3 = mm(&exp_so3([dc[s], dc[s + 1], dc[s + 2]]), &poses[i].r);
                    npose[i] = Pose { r, t: [poses[i].t[0] + dc[s + 3], poses[i].t[1] + dc[s + 4], poses[i].t[2] + dc[s + 5]] };
                }
            }
            let mut nk = *k;
            if cfg.intrinsics {
                nk.f += dc[kofs];
                nk.k1 += dc[kofs + 1];
                nk.k2 += dc[kofs + 2];
            }
            let ncost = total_cost(&nk, &npose, &np, obs, cfg.huber);
            if ncost < cost && nk.f > 0.0 {
                let gain = cost - ncost;
                *k = nk;
                poses.copy_from_slice(&npose);
                points.copy_from_slice(&np);
                cost = ncost;
                lambda = (lambda / 3.0).max(1e-9);
                accepted = true;
                if gain < 1e-10 * cost.max(1e-300) {
                    it = cfg.iters;
                }
                break;
            }
            lambda *= 5.0;
        }
        if !accepted {
            break;
        }
    }
    BaReport { rms_before, rms_after: rms(k, poses, points, obs), iterations: it }
}

fn mv3(m: &[f64; 9], v: [f64; 3]) -> [f64; 3] {
    [m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2], m[6] * v[0] + m[7] * v[1] + m[8] * v[2]]
}

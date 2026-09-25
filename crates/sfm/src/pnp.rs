// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Absolute pose from 2D-3D correspondences: the minimal three-point problem
//! (P3P, Grunert 1841; see Haralick et al., IJCV 1994) inside RANSAC, then
//! Gauss-Newton on the reprojection error of the inliers.
//!
//! P3P rather than the six-point DLT (Hartley & Zisserman §7.1): the DLT
//! estimates a full
//! 3x4 projection and is DEGENERATE when the points are coplanar - which is
//! what most of a capture of an object on a floor, a table or a deck is. It
//! also needs six clean points per sample against P3P's three, so at 30%
//! inliers it needs 27x more samples. Image points are normalized
//! coordinates.

use crate::camera::Pose;
use crate::linalg::{cholesky_solve, exp_so3, mm, mv, nearest_rotation, normalize, V3};
use data::rng::Lcg;

/// Every pose that puts world points `x` on the bearings of normalized image
/// points `u` exactly (up to four).
///
/// With depths `s_i` along unit bearings `j_i`, the law of cosines gives
/// `s_j² + s_k² - 2 s_j s_k cos(j_j, j_k) = |x_j - x_k|²` for each pair.
/// Substituting the depth ratios `u = s2/s1`, `v = s3/s1` and dividing two
/// of those by the third leaves, for each `v`, a quadratic in `u` and one
/// scalar residual in `v`; its roots are found by a log-spaced scan with
/// bisection, which is slower than a closed-form quartic and cannot silently
/// lose a root to a mis-transcribed coefficient.
pub fn p3p(x: [V3; 3], u: [[f64; 2]; 3]) -> Vec<Pose> {
    let j: [V3; 3] = std::array::from_fn(|i| normalize([u[i][0], u[i][1], 1.0]));
    let d2 = |a: V3, b: V3| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
    let (a2, b2, c2) = (d2(x[1], x[2]), d2(x[0], x[2]), d2(x[0], x[1]));
    if a2 < 1e-18 || b2 < 1e-18 || c2 < 1e-18 {
        return Vec::new();
    }
    let dot = |p: V3, q: V3| p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
    let (ca, cb, cg) = (dot(j[1], j[2]), dot(j[0], j[2]), dot(j[0], j[1]));
    // for a given v: the (up to two) u satisfying the c-equation, and the
    // a-equation's residual at each
    let branch = |v: f64, which: usize| -> Option<(f64, f64)> {
        let q = 1.0 + v * v - 2.0 * v * cb; // s1² = b² / q
        let k = 1.0 - (c2 / b2) * q; // u² - 2 cg u + k = 0
        let disc = cg * cg - k;
        if disc < 0.0 || q <= 0.0 {
            return None;
        }
        let uu = if which == 0 { cg + disc.sqrt() } else { cg - disc.sqrt() };
        if uu <= 0.0 {
            return None;
        }
        Some((uu, uu * uu + v * v - 2.0 * uu * v * ca - (a2 / b2) * q))
    };
    let mut roots: Vec<(f64, f64)> = Vec::new();
    for which in 0..2 {
        let mut prev: Option<(f64, f64)> = None;
        for step in 0..=3000 {
            let v = 1e-3 * (1e6f64).powf(step as f64 / 3000.0);
            let cur = branch(v, which).map(|(_, g)| (v, g));
            if let (Some((v0, g0)), Some((v1, g1))) = (prev, cur) {
                if g0 == 0.0 || g0.signum() != g1.signum() {
                    let (mut lo, mut hi, mut glo) = (v0, v1, g0);
                    for _ in 0..60 {
                        let mid = 0.5 * (lo + hi);
                        let Some((_, gm)) = branch(mid, which) else { break };
                        if gm.signum() == glo.signum() {
                            lo = mid;
                            glo = gm;
                        } else {
                            hi = mid;
                        }
                    }
                    let v = 0.5 * (lo + hi);
                    if let Some((uu, _)) = branch(v, which) {
                        roots.push((uu, v));
                    }
                }
            }
            prev = cur;
        }
    }
    let mut out = Vec::new();
    for (uu, v) in roots {
        let s1 = (b2 / (1.0 + v * v - 2.0 * v * cb)).sqrt();
        let depth = [s1, uu * s1, v * s1];
        let q: [V3; 3] = std::array::from_fn(|i| [j[i][0] * depth[i], j[i][1] * depth[i], j[i][2] * depth[i]]);
        // absolute orientation: q = R x + t
        let cx: V3 = std::array::from_fn(|k| (x[0][k] + x[1][k] + x[2][k]) / 3.0);
        let cq: V3 = std::array::from_fn(|k| (q[0][k] + q[1][k] + q[2][k]) / 3.0);
        let mut h = [0.0f64; 9];
        for i in 0..3 {
            for r in 0..3 {
                for c in 0..3 {
                    h[r * 3 + c] += (q[i][r] - cq[r]) * (x[i][c] - cx[c]);
                }
            }
        }
        let r = nearest_rotation(&h);
        let rc = mv(&r, cx);
        out.push(Pose { r, t: [cq[0] - rc[0], cq[1] - rc[1], cq[2] - rc[2]] });
    }
    out
}

/// Squared normalized reprojection error, infinite behind the camera.
pub fn residual2(p: &Pose, x: V3, u: [f64; 2]) -> f64 {
    let c = p.to_cam(x);
    if c[2] <= 1e-9 {
        return f64::INFINITY;
    }
    (c[0] / c[2] - u[0]).powi(2) + (c[1] / c[2] - u[1]).powi(2)
}

/// Gauss-Newton on the normalized reprojection error of `x`/`u`, left
/// perturbation on the rotation. A few iterations; stops when a step stops
/// helping.
pub fn refine(mut p: Pose, x: &[V3], u: &[[f64; 2]], iters: usize) -> Pose {
    let cost = |p: &Pose| -> f64 { x.iter().zip(u).map(|(a, b)| residual2(p, *a, *b).min(1e6)).sum() };
    let mut cur = cost(&p);
    for _ in 0..iters {
        let mut h = [0.0f64; 36];
        let mut g = [0.0f64; 6];
        for (xw, uv) in x.iter().zip(u) {
            let c = p.to_cam(*xw);
            if c[2] <= 1e-9 {
                continue;
            }
            let iz = 1.0 / c[2];
            let (px, py) = (c[0] * iz, c[1] * iz);
            let r = [px - uv[0], py - uv[1]];
            // d(proj)/d(c)
            let dp = [[iz, 0.0, -px * iz], [0.0, iz, -py * iz]];
            // c = exp(w) R X + t: dc/dw = -[R X]x, dc/dt = I
            let rx = [c[0] - p.t[0], c[1] - p.t[1], c[2] - p.t[2]];
            let dcdw = [[0.0, rx[2], -rx[1]], [-rx[2], 0.0, rx[0]], [rx[1], -rx[0], 0.0]];
            let mut j = [[0.0f64; 6]; 2];
            for row in 0..2 {
                for k in 0..3 {
                    j[row][k] = (0..3).map(|m| dp[row][m] * dcdw[m][k]).sum();
                    j[row][3 + k] = dp[row][k];
                }
            }
            for row in 0..2 {
                for a in 0..6 {
                    g[a] += j[row][a] * r[row];
                    for b in 0..6 {
                        h[a * 6 + b] += j[row][a] * j[row][b];
                    }
                }
            }
        }
        for a in 0..6 {
            h[a * 6 + a] *= 1.0 + 1e-6;
            h[a * 6 + a] += 1e-12;
        }
        let Some(d) = cholesky_solve(&h, &g.map(|v| -v), 6) else { break };
        let np = Pose { r: mm(&exp_so3([d[0], d[1], d[2]]), &p.r), t: [p.t[0] + d[3], p.t[1] + d[4], p.t[2] + d[5]] };
        let nc = cost(&np);
        if nc < cur {
            p = np;
            let done = cur - nc < 1e-12 * cur.max(1e-300);
            cur = nc;
            if done {
                break;
            }
        } else {
            break;
        }
    }
    p
}

/// RANSAC over P3P samples, then refinement on the consensus. `thresh` is in
/// normalized units. Returns the pose and the inlier mask.
pub fn ransac_pnp(x: &[V3], u: &[[f64; 2]], thresh: f64, iters: usize, seed: u64) -> Option<(Pose, Vec<bool>)> {
    let n = x.len();
    if n < 4 {
        return None;
    }
    let t2 = thresh * thresh;
    let mut rng = Lcg::new(seed);
    let mut best: Option<(Pose, usize)> = None;
    let mut need = iters;
    let mut done = 0;
    while done < need {
        done += 1;
        let mut idx = [0usize; 3];
        for k in 0..3 {
            loop {
                let c = (rng.next_u32() as usize) % n;
                if !idx[..k].contains(&c) {
                    idx[k] = c;
                    break;
                }
            }
        }
        for p in p3p([x[idx[0]], x[idx[1]], x[idx[2]]], [u[idx[0]], u[idx[1]], u[idx[2]]]) {
            let count = (0..n).filter(|&i| residual2(&p, x[i], u[i]) < t2).count();
            if best.as_ref().is_none_or(|(_, c)| count > *c) {
                best = Some((p, count));
                need = need.min(crate::ransac_iterations(count as f64 / n as f64, 3, iters));
            }
        }
    }
    let (mut p, _) = best?;
    let mut mask: Vec<bool> = (0..n).map(|i| residual2(&p, x[i], u[i]) < t2).collect();
    for _ in 0..3 {
        let (ix, iu): (Vec<V3>, Vec<[f64; 2]>) = (0..n).filter(|&i| mask[i]).map(|i| (x[i], u[i])).unzip();
        if ix.len() < 4 {
            return None;
        }
        p = refine(p, &ix, &iu, 20);
        mask = (0..n).map(|i| residual2(&p, x[i], u[i]) < t2).collect();
    }
    Some((p, mask))
}

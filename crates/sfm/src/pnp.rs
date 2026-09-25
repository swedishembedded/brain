// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Absolute pose from 2D-3D correspondences: the minimal three-point problem
//! (P3P, Grunert 1841; see Haralick et al., IJCV 1994) inside LO-RANSAC
//! (Chum, Matas & Kittler, DAGM 2003) with MSAC truncated-quadratic scoring,
//! then Gauss-Newton on the reprojection error of the inliers.
//!
//! P3P rather than the six-point DLT (Hartley & Zisserman §7.1): the DLT
//! estimates a full 3x4 projection and is DEGENERATE when the points are
//! coplanar - which is what most of a capture of an object on a floor, a
//! table or a deck is. It also needs six clean points per sample against
//! P3P's three, so at 30% inliers it needs 27x more samples.
//!
//! Image measurements are unit BEARINGS, so nothing assumes a point lies in
//! front of an image plane. The reprojection error of a bearing `f` is taken
//! on the plane tangent to the unit sphere at `f`: the camera-frame point
//! `c` lands at `(e₁·c, e₂·c) / (f·c)` in an orthonormal frame `(e₁, e₂, f)`,
//! the tangent of the angular error along each axis: for a central pinhole
//! ray that is the normalized-coordinate reprojection error, and it is
//! defined all the way round a fisheye's field.

use crate::camera::Pose;
use crate::linalg::{cholesky_solve, dot, exp_so3, mm, mv, nearest_rotation, normalize, V3};
use crate::twoview::tangent_basis;
use data::rng::Lcg;

/// Every pose that puts world points `x` on the unit bearings `u` exactly
/// (up to four).
///
/// With depths `s_i` along unit bearings `j_i`, the law of cosines gives
/// `s_j² + s_k² - 2 s_j s_k cos(j_j, j_k) = |x_j - x_k|²` for each pair.
/// Substituting the depth ratios `u = s2/s1`, `v = s3/s1` and dividing two
/// of those by the third leaves, for each `v`, a quadratic in `u` and one
/// scalar residual in `v`; its roots are found by a log-spaced scan with
/// bisection, which is slower than a closed-form quartic and cannot silently
/// lose a root to a mis-transcribed coefficient.
pub fn p3p(x: [V3; 3], u: [V3; 3]) -> Vec<Pose> {
    let j: [V3; 3] = u.map(normalize);
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

/// Squared tangent-plane reprojection error of world point `x` against unit
/// bearing `f` - `tan²` of the angle between them - infinite unless the
/// point lies ahead along the bearing.
pub fn residual2(p: &Pose, x: V3, f: V3) -> f64 {
    let c = p.to_cam(x);
    let d = dot(f, c);
    let c2 = dot(c, c);
    if d <= 1e-9 * c2.sqrt() {
        return f64::INFINITY;
    }
    ((c2 - d * d) / (d * d)).max(0.0)
}

/// Gauss-Newton on the tangent-plane reprojection error of `x`/`f`, left
/// perturbation on the rotation. Stops when a step stops helping.
pub fn refine(mut p: Pose, x: &[V3], f: &[V3], iters: usize) -> Pose {
    let basis: Vec<(V3, V3)> = f.iter().map(|b| tangent_basis(normalize(*b))).collect();
    let cost = |p: &Pose| -> f64 { x.iter().zip(f).map(|(a, b)| residual2(p, *a, *b).min(1e6)).sum() };
    let mut cur = cost(&p);
    for _ in 0..iters {
        let mut h = [0.0f64; 36];
        let mut g = [0.0f64; 6];
        for ((xw, fb), (e1, e2)) in x.iter().zip(f).zip(&basis) {
            let c = p.to_cam(*xw);
            let d = dot(*fb, c);
            if d <= 1e-9 {
                continue;
            }
            let r = [dot(*e1, c) / d, dot(*e2, c) / d];
            // d r / d c
            let dr: [V3; 2] = [std::array::from_fn(|k| (e1[k] - r[0] * fb[k]) / d), std::array::from_fn(|k| (e2[k] - r[1] * fb[k]) / d)];
            // c = exp(w) R X + t: dc/dw = -[R X]x, dc/dt = I
            let rx = [c[0] - p.t[0], c[1] - p.t[1], c[2] - p.t[2]];
            let dcdw = [[0.0, rx[2], -rx[1]], [-rx[2], 0.0, rx[0]], [rx[1], -rx[0], 0.0]];
            let mut j = [[0.0f64; 6]; 2];
            for row in 0..2 {
                for k in 0..3 {
                    j[row][k] = (0..3).map(|m| dr[row][m] * dcdw[m][k]).sum();
                    j[row][3 + k] = dr[row][k];
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

/// MSAC score (lower is better) and inlier count of `p` at squared
/// threshold `t2`.
fn msac(p: &Pose, x: &[V3], f: &[V3], t2: f64) -> (f64, usize) {
    let mut s = 0.0;
    let mut c = 0;
    for (a, b) in x.iter().zip(f) {
        let e = residual2(p, *a, *b);
        if e < t2 {
            s += e;
            c += 1;
        } else {
            s += t2;
        }
    }
    (s, c)
}

/// The correspondences `p` explains within squared threshold `t2`.
fn inliers(p: &Pose, x: &[V3], f: &[V3], t2: f64) -> (Vec<V3>, Vec<V3>) {
    x.iter().zip(f).filter(|(a, b)| residual2(p, **a, **b) < t2).map(|(a, b)| (*a, *b)).unzip()
}

/// Local optimization of a new best pose: Gauss-Newton on its inliers,
/// repeated while the inlier set grows (Chum et al.'s iterative least
/// squares, with the pose's own nonlinear refinement as the fit).
fn local_opt(p0: Pose, x: &[V3], f: &[V3], t2: f64) -> (Pose, f64, usize) {
    let (s0, c0) = msac(&p0, x, f, t2);
    let mut best = (p0, s0, c0);
    for _ in 0..4 {
        let (ix, iu) = inliers(&best.0, x, f, t2);
        if ix.len() < 4 {
            break;
        }
        let p = refine(best.0, &ix, &iu, 5);
        let (s, c) = msac(&p, x, f, t2);
        if s >= best.1 {
            break;
        }
        best = (p, s, c);
    }
    best
}

/// LO-RANSAC over P3P samples with MSAC scoring, then Gauss-Newton on the
/// consensus. `thresh` is the angular inlier threshold in radians (a pixel
/// threshold divided by the focal length). Returns the pose and the inlier
/// mask.
pub fn ransac_pnp(x: &[V3], f: &[V3], thresh: f64, iters: usize, seed: u64) -> Option<(Pose, Vec<bool>)> {
    let n = x.len();
    if n < 4 {
        return None;
    }
    let t2 = thresh * thresh;
    let mut rng = Lcg::new(seed);
    let mut best: Option<(Pose, f64, usize)> = None;
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
        for p in p3p([x[idx[0]], x[idx[1]], x[idx[2]]], [f[idx[0]], f[idx[1]], f[idx[2]]]) {
            let (s, _) = msac(&p, x, f, t2);
            if best.as_ref().is_none_or(|b| s < b.1) {
                let lo = local_opt(p, x, f, t2);
                need = need.min(crate::ransac_iterations(lo.2 as f64 / n as f64, 3, iters));
                best = Some(lo);
            }
        }
    }
    let (mut p, _, _) = best?;
    for _ in 0..3 {
        let (ix, iu) = inliers(&p, x, f, t2);
        if ix.len() < 4 {
            return None;
        }
        p = refine(p, &ix, &iu, 20);
    }
    let mask = (0..n).map(|i| residual2(&p, x[i], f[i]) < t2).collect();
    Some((p, mask))
}

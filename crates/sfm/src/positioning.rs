// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Global positioning: every camera centre and every point at once, from
//! the cameras' rotations and the bearings in which they see the points -
//! no pairwise translations, no incremental chain for error to accumulate
//! along.
//!
//! The formulation is GLOMAP's (Pan et al., "Global Structure-from-Motion
//! Revisited", ECCV 2024), itself the BATA objective (Zhuang et al.,
//! "Baseline Desensitizing In Translation Averaging", CVPR 2018) applied to
//! camera-to-point constraints:
//!
//! ```text
//! minimize  sum_ij  rho( | v_ij - d_ij (X_j - c_i) | )   over c, X and d_ij >= 0
//! ```
//!
//! where `v_ij` is the unit world-frame bearing from camera `i` to point `j`
//! (its measured ray turned by its averaged rotation). Each residual is at
//! most 1 whatever the distances involved, so a far point or a wrong match
//! pulls with bounded force, and the free per-observation scale `d_ij` makes
//! the cost blind to depth: it measures only whether the point lies ALONG
//! the ray, the one thing a bearing says.
//!
//! The solve starts from random positions, as GLOMAP's does (the objective
//! is benign enough that no initialization from pairwise geometry is needed),
//! and alternates exactly-solved blocks, each of which can only lower the
//! cost: every `d_ij` in closed form (`max(0, v.w / |w|^2)`), then every
//! position by reweighted linear least squares under a Huber loss. With the
//! scales fixed the position problem is `X_j - c_i ~ v_ij / d_ij` with
//! weights `d_ij^2`, whose normal equations are a multiple of the identity
//! per coordinate; the points are eliminated in closed form and the camera
//! system, one small matrix shared by the three coordinates, is solved
//! directly. That is a departure from GLOMAP, which minimizes jointly with a
//! general nonlinear solver; the alternation reaches the same stationary
//! points without one.
//!
//! The gauge: the first camera sits at the origin and the median
//! camera-to-point distance is 1. Rotations are world-to-camera.
//!
//! Swedish Embedded AB implements global structure from motion for its
//! clients. If your team needs large photo collections turned into
//! calibrated cameras, you can procure our services by sending an email to
//! info@swedishembedded.com.

use crate::linalg::{add, cholesky_solve, dot, mtv, norm, normalize, scale, sub, M3, V3};
use data::rng::Lcg;

/// Camera `cam` sees point `point` along `ray`, a unit bearing in the
/// camera's own frame.
#[derive(Clone, Copy, Debug)]
pub struct RayObservation {
    pub cam: usize,
    pub point: usize,
    pub ray: V3,
}

#[derive(Clone, Copy, Debug)]
pub struct PositioningCfg {
    /// Most alternation rounds.
    pub iters: usize,
    /// Huber threshold on the residual, which is about the sine of the
    /// angle by which a point misses its ray.
    pub huber: f64,
    /// Seed of the random start.
    pub seed: u64,
}

impl Default for PositioningCfg {
    fn default() -> Self {
        PositioningCfg { iters: 400, huber: 0.01, seed: 1 }
    }
}

#[derive(Clone, Debug)]
pub struct Positions {
    /// Per camera, `None` for one that no usable observation constrains.
    pub centres: Vec<Option<V3>>,
    /// Per point, `None` likewise.
    pub points: Vec<Option<V3>>,
    /// Per observation, whether the result puts its point along its ray
    /// (ahead of the camera, within the Huber threshold).
    pub inlier: Vec<bool>,
}

fn huber_weight(r: f64, delta: f64) -> f64 {
    if r <= delta { 1.0 } else { delta / r }
}

fn huber(r: f64, delta: f64) -> f64 {
    if r <= delta { r * r } else { 2.0 * delta * r - delta * delta }
}

/// Place `rots.len()` cameras and `npoints` points from `obs`.
pub fn global_positions(rots: &[M3], npoints: usize, obs: &[RayObservation], cfg: &PositioningCfg) -> Positions {
    let ncam = rots.len();
    let v: Vec<V3> = obs.iter().map(|o| normalize(mtv(&rots[o.cam], o.ray))).collect();
    let mut rng = Lcg::new(cfg.seed);
    let mut rnd = || [rng.signed() as f64, rng.signed() as f64, rng.signed() as f64];
    let mut c: Vec<V3> = (0..ncam).map(|_| rnd()).collect();
    let mut x: Vec<V3> = (0..npoints).map(|_| rnd()).collect();
    let mut by_point: Vec<Vec<usize>> = vec![Vec::new(); npoints];
    for (i, o) in obs.iter().enumerate() {
        by_point[o.point].push(i);
    }
    let mut d = vec![0.0f64; obs.len()];
    let mut cam_used = vec![false; ncam];
    let mut point_used = vec![false; npoints];
    let mut last = f64::INFINITY;
    let mut still = 0;
    for _ in 0..cfg.iters {
        // ---- scales, in closed form, and the reweighting ----
        let mut a = vec![0.0f64; obs.len()];
        let mut cost = 0.0;
        for (i, o) in obs.iter().enumerate() {
            let w = sub(x[o.point], c[o.cam]);
            let ww = dot(w, w);
            d[i] = if ww > 1e-300 { (dot(v[i], w) / ww).max(0.0) } else { 0.0 };
            let r = norm(sub(v[i], scale(w, d[i])));
            cost += huber(r, cfg.huber);
            a[i] = if d[i] > 0.0 { huber_weight(r, cfg.huber) * d[i] * d[i] } else { 0.0 };
        }
        if cost > last * (1.0 - 1e-9) {
            still += 1;
            if still >= 3 {
                break;
            }
        } else {
            still = 0;
        }
        last = last.min(cost);
        // ---- positions: X_j - c_i ~ g, weight a, g = v / d ----
        cam_used.iter_mut().for_each(|u| *u = false);
        let mut asum = vec![0.0f64; npoints];
        let mut ag = vec![[0.0f64; 3]; npoints];
        for (i, o) in obs.iter().enumerate() {
            if a[i] > 0.0 {
                cam_used[o.cam] = true;
                asum[o.point] += a[i];
                ag[o.point] = add(ag[o.point], scale(v[i], a[i] / d[i]));
            }
        }
        let Some(gauge) = (0..ncam).find(|&k| cam_used[k]) else { break };
        let mut col = vec![usize::MAX; ncam];
        let mut m = 0usize;
        for k in 0..ncam {
            if cam_used[k] && k != gauge {
                col[k] = m;
                m += 1;
            }
        }
        let mut s = vec![0.0f64; m * m];
        let mut rhs = vec![[0.0f64; 3]; m];
        for (j, list) in by_point.iter().enumerate() {
            if asum[j] <= 0.0 {
                continue;
            }
            for &oi in list {
                if a[oi] <= 0.0 {
                    continue;
                }
                let ci = col[obs[oi].cam];
                if ci == usize::MAX {
                    continue;
                }
                s[ci * m + ci] += a[oi];
                let gi = scale(v[oi], a[oi] / d[oi]);
                rhs[ci] = sub(rhs[ci], gi);
                rhs[ci] = add(rhs[ci], scale(ag[j], a[oi] / asum[j]));
                for &ok in list {
                    let ck = col[obs[ok].cam];
                    if a[ok] > 0.0 && ck != usize::MAX {
                        s[ci * m + ck] -= a[oi] * a[ok] / asum[j];
                    }
                }
            }
        }
        // a tiny ridge keeps a camera tied to the rest by one point solvable
        for k in 0..m {
            s[k * m + k] += 1e-12 * s[k * m + k].abs().max(1e-12);
        }
        let mut solved = vec![[0.0f64; 3]; m];
        let mut ok = true;
        for axis in 0..3 {
            let b: Vec<f64> = rhs.iter().map(|r| r[axis]).collect();
            match cholesky_solve(&s, &b, m) {
                Some(sol) => sol.iter().enumerate().for_each(|(k, v)| solved[k][axis] = *v),
                None => ok = false,
            }
        }
        if !ok {
            break;
        }
        c[gauge] = [0.0; 3];
        for k in 0..ncam {
            if col[k] != usize::MAX {
                c[k] = solved[col[k]];
            }
        }
        for (j, list) in by_point.iter().enumerate() {
            point_used[j] = asum[j] > 0.0;
            if !point_used[j] {
                continue;
            }
            let mut acc = ag[j];
            for &oi in list {
                if a[oi] > 0.0 {
                    acc = add(acc, scale(c[obs[oi].cam], a[oi]));
                }
            }
            x[j] = scale(acc, 1.0 / asum[j]);
        }
        // ---- the scale gauge: median camera-to-point distance 1 ----
        let mut dist: Vec<f64> = obs.iter().filter(|o| cam_used[o.cam] && point_used[o.point]).map(|o| norm(sub(x[o.point], c[o.cam]))).collect();
        if dist.is_empty() {
            break;
        }
        let mid = dist.len() / 2;
        let med = *dist.select_nth_unstable_by(mid, f64::total_cmp).1;
        if med > 0.0 && med.is_finite() {
            c.iter_mut().for_each(|p| *p = scale(*p, 1.0 / med));
            x.iter_mut().for_each(|p| *p = scale(*p, 1.0 / med));
        }
    }
    let inlier: Vec<bool> = obs
        .iter()
        .enumerate()
        .map(|(i, o)| {
            let w = sub(x[o.point], c[o.cam]);
            let ww = dot(w, w);
            let di = if ww > 1e-300 { (dot(v[i], w) / ww).max(0.0) } else { 0.0 };
            di > 0.0 && norm(sub(v[i], scale(w, di))) <= cfg.huber
        })
        .collect();
    Positions {
        centres: c.iter().zip(&cam_used).map(|(p, &u)| u.then_some(*p)).collect(),
        points: x.iter().zip(&point_used).map(|(p, &u)| u.then_some(*p)).collect(),
        inlier,
    }
}

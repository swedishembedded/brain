// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Absolute pose from 2D-3D correspondences: the calibrated direct linear
//! transform (Hartley & Zisserman §7.1) on six-point samples inside RANSAC,
//! then Gauss-Newton on the reprojection error of the inliers.
//!
//! Image points are normalized coordinates, so the DLT estimates `[R|t]` up to
//! scale directly; the rotation block is projected onto SO(3) and the scale
//! read off its singular values.

use crate::camera::Pose;
use crate::linalg::{cholesky_solve, exp_so3, mm, mv, nearest_rotation, null_vector, svd3, M3, V3};
use data::rng::Lcg;

fn dlt(x: &[V3], u: &[[f64; 2]]) -> Option<Pose> {
    if x.len() < 6 {
        return None;
    }
    // condition the world points: centre and scale to unit RMS
    let n = x.len() as f64;
    let c: V3 = std::array::from_fn(|k| x.iter().map(|p| p[k]).sum::<f64>() / n);
    let rms = (x.iter().map(|p| (0..3).map(|k| (p[k] - c[k]).powi(2)).sum::<f64>()).sum::<f64>() / n).sqrt().max(1e-12);
    let s = 1.0 / rms;
    let mut rows = Vec::with_capacity(2 * x.len());
    for (p, q) in x.iter().zip(u) {
        let (a, b, cc) = ((p[0] - c[0]) * s, (p[1] - c[1]) * s, (p[2] - c[2]) * s);
        rows.push(vec![a, b, cc, 1.0, 0.0, 0.0, 0.0, 0.0, -q[0] * a, -q[0] * b, -q[0] * cc, -q[0]]);
        rows.push(vec![0.0, 0.0, 0.0, 0.0, a, b, cc, 1.0, -q[1] * a, -q[1] * b, -q[1] * cc, -q[1]]);
    }
    let h = null_vector(&rows, 12);
    let m: M3 = [h[0], h[1], h[2], h[4], h[5], h[6], h[8], h[9], h[10]];
    let (_, sv, _) = svd3(&m);
    let scale = (sv[0] + sv[1] + sv[2]) / 3.0;
    if scale < 1e-12 {
        return None;
    }
    let mut sign = 1.0;
    // the recovered projection must put the points in front
    let t0 = [h[3] / scale, h[7] / scale, h[11] / scale];
    let front = x.iter().filter(|p| {
        let d = (m[6] * (p[0] - c[0]) * s + m[7] * (p[1] - c[1]) * s + m[8] * (p[2] - c[2]) * s) / scale + t0[2];
        d > 0.0
    }).count();
    if front * 2 < x.len() {
        sign = -1.0;
    }
    let r = nearest_rotation(&m.map(|v| sign * v / scale));
    let tn = [sign * h[3] / scale, sign * h[7] / scale, sign * h[11] / scale];
    // undo the conditioning: X' = s (X - c)  =>  t = tn - s R c ... with the
    // scale folded back: x_cam = R s (X - c) + tn, divide by s
    let rc = mv(&r, c);
    let t = [tn[0] / s - rc[0], tn[1] / s - rc[1], tn[2] / s - rc[2]];
    Some(Pose { r, t })
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

/// RANSAC over six-point DLT samples, then refinement on the consensus.
/// `thresh` is in normalized units. Returns the pose and the inlier mask.
pub fn ransac_pnp(x: &[V3], u: &[[f64; 2]], thresh: f64, iters: usize, seed: u64) -> Option<(Pose, Vec<bool>)> {
    let n = x.len();
    if n < 6 {
        return None;
    }
    let t2 = thresh * thresh;
    let mut rng = Lcg::new(seed);
    let mut best: Option<(Pose, usize)> = None;
    for _ in 0..iters {
        let mut idx = [0usize; 6];
        for k in 0..6 {
            loop {
                let c = (rng.next_u32() as usize) % n;
                if !idx[..k].contains(&c) {
                    idx[k] = c;
                    break;
                }
            }
        }
        let sx: Vec<V3> = idx.iter().map(|&i| x[i]).collect();
        let su: Vec<[f64; 2]> = idx.iter().map(|&i| u[i]).collect();
        let Some(p) = dlt(&sx, &su) else { continue };
        let count = (0..n).filter(|&i| residual2(&p, x[i], u[i]) < t2).count();
        if best.as_ref().is_none_or(|(_, c)| count > *c) {
            best = Some((p, count));
        }
    }
    let (mut p, _) = best?;
    let mut mask: Vec<bool> = (0..n).map(|i| residual2(&p, x[i], u[i]) < t2).collect();
    for _ in 0..3 {
        let (ix, iu): (Vec<V3>, Vec<[f64; 2]>) = (0..n).filter(|&i| mask[i]).map(|i| (x[i], u[i])).unzip();
        if ix.len() < 6 {
            return None;
        }
        p = refine(p, &ix, &iu, 20);
        mask = (0..n).map(|i| residual2(&p, x[i], u[i]) < t2).collect();
    }
    Some((p, mask))
}

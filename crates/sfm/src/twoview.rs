// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Calibrated two-view geometry: the essential matrix by the normalized
//! eight-point algorithm inside RANSAC (Hartley & Zisserman, "Multiple View
//! Geometry", §9.6 and §11.2; Fischler & Bolles 1981), its decomposition into
//! the four rigid motions and the cheirality test that picks one (§9.6.2),
//! and linear (DLT) triangulation (§12.2).
//!
//! All image points here are NORMALIZED coordinates (`K⁻¹x`, undistorted), so
//! the eight-point system is already well conditioned and inlier thresholds
//! are angles rather than pixels.

use crate::camera::Pose;
use crate::linalg::{dot, mm, norm, normalize, null_vector, sub, svd3, transpose, M3, V3};
use data::rng::Lcg;

/// Sampson distance of a correspondence to the epipolar geometry of `e`, in
/// normalized units (the first-order geometric error, H&Z §11.4.3).
pub fn sampson(e: &M3, a: [f64; 2], b: [f64; 2]) -> f64 {
    let x1 = [a[0], a[1], 1.0];
    let x2 = [b[0], b[1], 1.0];
    let ex1 = crate::linalg::mv(e, x1);
    let etx2 = crate::linalg::mtv(e, x2);
    let num = dot(x2, ex1);
    let den = ex1[0] * ex1[0] + ex1[1] * ex1[1] + etx2[0] * etx2[0] + etx2[1] * etx2[1];
    num * num / den.max(1e-300)
}

/// The essential matrix of at least eight correspondences, projected onto the
/// essential manifold (two equal singular values, one zero).
pub fn essential_8pt(a: &[[f64; 2]], b: &[[f64; 2]]) -> Option<M3> {
    if a.len() < 8 {
        return None;
    }
    let rows: Vec<Vec<f64>> = a
        .iter()
        .zip(b)
        .map(|(p, q)| vec![q[0] * p[0], q[0] * p[1], q[0], q[1] * p[0], q[1] * p[1], q[1], p[0], p[1], 1.0])
        .collect();
    let e = null_vector(&rows, 9);
    let e: M3 = std::array::from_fn(|i| e[i]);
    let (u, s, v) = svd3(&e);
    if s[0] <= 0.0 {
        return None;
    }
    let d = [1.0, 1.0, 0.0];
    let ud: M3 = std::array::from_fn(|i| u[i] * d[i % 3]);
    Some(mm(&ud, &transpose(&v)))
}

/// RANSAC over [`essential_8pt`]. `thresh` is the Sampson threshold in
/// normalized units (a pixel threshold divided by the focal length).
/// Returns the essential matrix refit on all inliers, and the inlier mask.
pub fn ransac_essential(a: &[[f64; 2]], b: &[[f64; 2]], thresh: f64, iters: usize, seed: u64) -> Option<(M3, Vec<bool>)> {
    let n = a.len();
    if n < 8 {
        return None;
    }
    let t2 = thresh * thresh;
    let mut rng = Lcg::new(seed);
    let mut best: Option<(M3, usize)> = None;
    let mut sa = [[0.0; 2]; 8];
    let mut sb = [[0.0; 2]; 8];
    let mut need = iters;
    let mut done = 0;
    while done < need {
        done += 1;
        let mut idx = [0usize; 8];
        for k in 0..8 {
            loop {
                let c = (rng.next_u32() as usize) % n;
                if !idx[..k].contains(&c) {
                    idx[k] = c;
                    break;
                }
            }
            sa[k] = a[idx[k]];
            sb[k] = b[idx[k]];
        }
        let Some(e) = essential_8pt(&sa, &sb) else { continue };
        let count = (0..n).filter(|&i| sampson(&e, a[i], b[i]) < t2).count();
        if best.as_ref().is_none_or(|(_, c)| count > *c) {
            best = Some((e, count));
            need = need.min(crate::ransac_iterations(count as f64 / n as f64, 8, iters));
        }
    }
    let (e, _) = best?;
    // refit on the consensus, twice, so the final model is the least-squares
    // answer of its own inliers rather than of eight lucky points
    let mut e = e;
    let mut mask: Vec<bool> = (0..n).map(|i| sampson(&e, a[i], b[i]) < t2).collect();
    for _ in 0..2 {
        let (ia, ib): (Vec<[f64; 2]>, Vec<[f64; 2]>) = (0..n).filter(|&i| mask[i]).map(|i| (a[i], b[i])).unzip();
        if let Some(e2) = essential_8pt(&ia, &ib) {
            let m2: Vec<bool> = (0..n).map(|i| sampson(&e2, a[i], b[i]) < t2).collect();
            if m2.iter().filter(|&&v| v).count() >= mask.iter().filter(|&&v| v).count() {
                e = e2;
                mask = m2;
            }
        }
    }
    Some((e, mask))
}

/// Linear triangulation of one point seen in several views: the DLT over
/// `[x]ₓ P X = 0`, with each view's rows scaled to unit weight.
pub fn triangulate(poses: &[&Pose], pts: &[[f64; 2]]) -> Option<V3> {
    let mut rows = Vec::with_capacity(poses.len() * 2);
    for (p, x) in poses.iter().zip(pts) {
        let r = &p.r;
        let row = |k: usize| [r[k * 3], r[k * 3 + 1], r[k * 3 + 2], p.t[k]];
        let (p0, p1, p2) = (row(0), row(1), row(2));
        rows.push((0..4).map(|j| x[0] * p2[j] - p0[j]).collect::<Vec<f64>>());
        rows.push((0..4).map(|j| x[1] * p2[j] - p1[j]).collect::<Vec<f64>>());
    }
    let h = null_vector(&rows, 4);
    if h[3].abs() < 1e-12 {
        return None;
    }
    Some([h[0] / h[3], h[1] / h[3], h[2] / h[3]])
}

/// The angle in radians between the rays from two camera centres to `x`.
pub fn ray_angle(c1: V3, c2: V3, x: V3) -> f64 {
    let (a, b) = (normalize(sub(x, c1)), normalize(sub(x, c2)));
    dot(a, b).clamp(-1.0, 1.0).acos()
}

/// The relative pose `(R, t)` (second camera, first at identity, `|t| = 1`)
/// of the four decompositions of `e` that puts the most correspondences in
/// front of both cameras. Returns the pose and the triangulated points of the
/// correspondences in `mask` (`None` for ones that did not triangulate in
/// front).
pub fn relative_pose(e: &M3, a: &[[f64; 2]], b: &[[f64; 2]], mask: &[bool]) -> (Pose, Vec<Option<V3>>) {
    let (mut u, _, mut v) = svd3(e);
    if crate::linalg::det(&u) < 0.0 {
        u = u.map(|x| -x);
    }
    if crate::linalg::det(&v) < 0.0 {
        v = v.map(|x| -x);
    }
    let w: M3 = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let r1 = mm(&mm(&u, &w), &transpose(&v));
    let r2 = mm(&mm(&u, &transpose(&w)), &transpose(&v));
    let t = [u[2], u[5], u[8]];
    let first = Pose::identity();
    let mut best: Option<(usize, Pose, Vec<Option<V3>>)> = None;
    for (r, t) in [(r1, t), (r1, t.map(|x| -x)), (r2, t), (r2, t.map(|x| -x))] {
        let second = Pose { r, t };
        let mut pts = Vec::with_capacity(a.len());
        let mut good = 0usize;
        for i in 0..a.len() {
            if !mask[i] {
                pts.push(None);
                continue;
            }
            let x = triangulate(&[&first, &second], &[a[i], b[i]]);
            let ok = x.filter(|x| x[2] > 0.0 && second.to_cam(*x)[2] > 0.0);
            if ok.is_some() {
                good += 1;
            }
            pts.push(ok);
        }
        if best.as_ref().is_none_or(|(g, _, _)| good > *g) {
            best = Some((good, second, pts));
        }
    }
    let (_, p, pts) = best.unwrap();
    (p, pts)
}

/// Median angle between the two rays of the triangulated points - the
/// baseline measure that decides whether a pair can seed a reconstruction.
pub fn median_angle(second: &Pose, pts: &[Option<V3>]) -> f64 {
    let (c1, c2) = ([0.0; 3], second.centre());
    let mut a: Vec<f64> = pts.iter().flatten().map(|x| ray_angle(c1, c2, *x)).collect();
    if a.is_empty() {
        return 0.0;
    }
    a.sort_by(f64::total_cmp);
    a[a.len() / 2]
}

/// `[t]ₓ R`, the essential matrix of a relative pose.
pub fn essential_of(p: &Pose) -> M3 {
    let t = if norm(p.t) > 0.0 { normalize(p.t) } else { p.t };
    let tx = crate::linalg::skew(t);
    mm(&tx, &p.r)
}

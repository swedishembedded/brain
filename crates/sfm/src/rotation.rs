// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Robust rotation averaging: every camera's orientation from the relative
//! rotations of the verified image pairs (the view graph), some of which are
//! wrong - a repeated texture or a symmetric object verifies an essential
//! matrix that is not the scene's.
//!
//! Implemented from Chatterjee & Govindu, "Robust Relative Rotation
//! Averaging" (TPAMI 2018): an initial estimate is propagated along the
//! maximum-weight spanning tree of the view graph (the pairs with the most
//! verified inliers first), then refined on the Lie algebra by iteratively
//! reweighted least squares - first under an L1 cost, whose basin is wide,
//! then under Geman-McClure, which gives a wrong relative rotation almost no
//! pull at all. Each step solves the linearized system
//!
//! ```text
//! minimize  sum_e  w_e | m_e + R_ab w_a - w_b |^2 ,   m_e = log(R_ab R_a R_b^T)
//! ```
//!
//! for left increments `R_i <- exp(w_i) R_i`, one camera held (the gauge).
//! Because every `R_ab` is orthonormal, the system's diagonal blocks are
//! multiples of the identity, and it is solved matrix-free by conjugate
//! gradients with that diagonal as preconditioner - linear in the number of
//! pairs, so a view graph of thousands of images costs what its edges cost.
//!
//! A pair whose relative rotation the averaged rotations miss by more than
//! [`RotationCfg::max_error_deg`] is reported as an outlier, and a camera
//! reached only through outliers is left without a rotation.
//!
//! Conventions: rotations are world-to-camera, `R_ab = R_b R_a^T` takes
//! camera `a`'s frame to camera `b`'s - what [`crate::twoview::relative_pose`]
//! returns for the second camera with the first at the identity.
//!
//! Swedish Embedded AB implements global structure from motion for its
//! clients. If your team needs large photo collections turned into
//! calibrated cameras, you can procure our services by sending an email to
//! info@swedishembedded.com.

use crate::linalg::{add, exp_so3, log_so3, mm, mtv, mv, norm, scale, sub, transpose, M3, V3};

/// One edge of the view graph.
#[derive(Clone, Copy, Debug)]
pub struct RelativeRotation {
    pub a: usize,
    pub b: usize,
    /// `R_b R_a^T`.
    pub r: M3,
    /// Confidence, e.g. verified inliers; only ratios matter.
    pub weight: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct RotationCfg {
    /// A relative rotation the result misses by more than this is an
    /// outlier; also the Geman-McClure scale.
    pub max_error_deg: f64,
    /// Reweighting rounds per phase.
    pub iters: usize,
}

impl Default for RotationCfg {
    fn default() -> Self {
        RotationCfg { max_error_deg: 5.0, iters: 40 }
    }
}

/// The averaged rotations.
#[derive(Clone, Debug)]
pub struct Rotations {
    /// Per camera, `None` where no inlier path reaches it from the largest
    /// connected part of the view graph.
    pub rotations: Vec<Option<M3>>,
    /// Per edge, whether the result agrees with it.
    pub inlier: Vec<bool>,
}

fn find(p: &mut [usize], mut x: usize) -> usize {
    while p[x] != x {
        p[x] = p[p[x]];
        x = p[x];
    }
    x
}

/// Every camera reachable from `root` over `edges` for which `keep` holds,
/// with a rotation propagated along the first edge that reached it.
fn propagate(n: usize, edges: &[RelativeRotation], keep: &[bool], root: usize) -> Vec<Option<M3>> {
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, e) in edges.iter().enumerate() {
        if keep[i] {
            adj[e.a].push(i);
            adj[e.b].push(i);
        }
    }
    let mut rot = vec![None; n];
    rot[root] = Some(crate::linalg::I3);
    let mut queue = std::collections::VecDeque::from([root]);
    while let Some(v) = queue.pop_front() {
        let rv = rot[v].unwrap();
        for &i in &adj[v] {
            let e = &edges[i];
            let (other, r) = if e.a == v { (e.b, mm(&e.r, &rv)) } else { (e.a, mm(&transpose(&e.r), &rv)) };
            if rot[other].is_none() {
                rot[other] = Some(r);
                queue.push_back(other);
            }
        }
    }
    rot
}

/// `m_e`, the rotation by which the current estimate misses edge `e`,
/// in camera `b`'s frame.
fn miss(e: &RelativeRotation, rot: &[Option<M3>]) -> Option<V3> {
    let (ra, rb) = (rot[e.a]?, rot[e.b]?);
    Some(log_so3(&mm(&mm(&e.r, &ra), &transpose(&rb))))
}

/// Average the relative rotations `edges` over `n` cameras.
pub fn average_rotations(n: usize, edges: &[RelativeRotation], cfg: &RotationCfg) -> Rotations {
    if n == 0 || edges.is_empty() {
        return Rotations { rotations: vec![None; n], inlier: vec![false; edges.len()] };
    }
    // ---- maximum-weight spanning forest (Kruskal), its largest tree ----
    let mut order: Vec<usize> = (0..edges.len()).collect();
    order.sort_by(|&i, &j| edges[j].weight.total_cmp(&edges[i].weight).then(i.cmp(&j)));
    let mut parent: Vec<usize> = (0..n).collect();
    let mut tree = vec![false; edges.len()];
    for &i in &order {
        let (x, y) = (find(&mut parent, edges[i].a), find(&mut parent, edges[i].b));
        if x != y {
            parent[x] = y;
            tree[i] = true;
        }
    }
    let comp: Vec<usize> = (0..n).map(|v| find(&mut parent, v)).collect();
    let mut size = vec![0usize; n];
    for &c in &comp {
        size[c] += 1;
    }
    let mut strength = vec![0.0f64; n];
    for e in edges {
        strength[e.a] += e.weight;
        strength[e.b] += e.weight;
    }
    let big = (0..n).max_by_key(|&v| (size[comp[v]], std::cmp::Reverse(v))).map(|v| comp[v]).unwrap();
    // the gauge: the best-connected camera of the largest part
    let root = (0..n).filter(|&v| comp[v] == big).max_by(|&a, &b| strength[a].total_cmp(&strength[b]).then(b.cmp(&a))).unwrap();
    let mut rot = propagate(n, edges, &tree, root);

    // ---- IRLS on the Lie algebra: L1, then Geman-McClure ----
    let mean_w = edges.iter().map(|e| e.weight).sum::<f64>() / edges.len() as f64;
    let sigma = cfg.max_error_deg.to_radians();
    for phase in 0..2 {
        for _ in 0..cfg.iters {
            let mut sys: Vec<(usize, usize, M3, V3, f64)> = Vec::with_capacity(edges.len());
            for e in edges {
                let Some(m) = miss(e, &rot) else { continue };
                let r = norm(m);
                let robust = if phase == 0 { 1.0 / r.max(1e-4) } else { (sigma * sigma / (sigma * sigma + r * r)).powi(2) };
                sys.push((e.a, e.b, e.r, m, robust * e.weight / mean_w));
            }
            let step = solve(n, root, &rot, &sys);
            let mut biggest = 0.0f64;
            for (v, w) in step.iter().enumerate() {
                if let (Some(r), Some(w)) = (rot[v], w) {
                    rot[v] = Some(mm(&exp_so3(*w), &r));
                    biggest = biggest.max(norm(*w));
                }
            }
            if biggest < 1e-9 {
                break;
            }
        }
    }

    let inlier: Vec<bool> = edges.iter().map(|e| miss(e, &rot).is_some_and(|m| norm(m) < sigma)).collect();
    let reach = propagate(n, edges, &inlier, root);
    let rotations = rot.iter().zip(&reach).map(|(r, k)| if k.is_some() { *r } else { None }).collect();
    Rotations { rotations, inlier }
}

/// Solve the linearized, reweighted system `sum w |m + R_ab x_a - x_b|^2`
/// with `x_root = 0` by preconditioned conjugate gradients.
fn solve(n: usize, root: usize, rot: &[Option<M3>], sys: &[(usize, usize, M3, V3, f64)]) -> Vec<Option<V3>> {
    let active: Vec<bool> = (0..n).map(|v| v != root && rot[v].is_some()).collect();
    // H x: diagonal blocks are (sum of w) I because R_ab^T R_ab = I
    let apply = |x: &[V3]| -> Vec<V3> {
        let mut y = vec![[0.0; 3]; n];
        for &(a, b, r, _, w) in sys {
            let (xa, xb) = (x[a], x[b]);
            // residual direction R xa - xb, its gradient: a gets R^T(.), b -(.)
            let d = sub(mv(&r, xa), xb);
            y[a] = add(y[a], scale(mtv(&r, d), w));
            y[b] = sub(y[b], scale(d, w));
        }
        for (v, yv) in y.iter_mut().enumerate() {
            if !active[v] {
                *yv = [0.0; 3];
            }
        }
        y
    };
    let mut diag = vec![0.0f64; n];
    let mut rhs = vec![[0.0f64; 3]; n];
    for &(a, b, r, m, w) in sys {
        diag[a] += w;
        diag[b] += w;
        rhs[a] = sub(rhs[a], scale(mtv(&r, m), w));
        rhs[b] = add(rhs[b], scale(m, w));
    }
    for (v, r) in rhs.iter_mut().enumerate() {
        if !active[v] {
            *r = [0.0; 3];
        }
    }
    let precond = |r: &[V3]| -> Vec<V3> { r.iter().zip(&diag).map(|(v, &d)| if d > 0.0 { scale(*v, 1.0 / d) } else { [0.0; 3] }).collect() };
    let dotv = |a: &[V3], b: &[V3]| -> f64 { a.iter().zip(b).map(|(x, y)| x[0] * y[0] + x[1] * y[1] + x[2] * y[2]).sum() };
    let mut x = vec![[0.0f64; 3]; n];
    let mut r = rhs.clone();
    let mut z = precond(&r);
    let mut p = z.clone();
    let mut rz = dotv(&r, &z);
    let r0 = dotv(&r, &r).sqrt();
    for _ in 0..(6 * n).max(50) {
        if dotv(&r, &r).sqrt() <= 1e-12 * r0.max(1e-300) {
            break;
        }
        let hp = apply(&p);
        let php = dotv(&p, &hp);
        if php <= 0.0 {
            break;
        }
        let alpha = rz / php;
        for v in 0..n {
            x[v] = add(x[v], scale(p[v], alpha));
            r[v] = sub(r[v], scale(hp[v], alpha));
        }
        z = precond(&r);
        let rz_new = dotv(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;
        for v in 0..n {
            p[v] = add(z[v], scale(p[v], beta));
        }
    }
    (0..n).map(|v| active[v].then_some(x[v])).collect()
}

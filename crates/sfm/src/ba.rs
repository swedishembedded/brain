// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Bundle adjustment: Levenberg-Marquardt on the pixel reprojection error of
//! every observation, over every camera pose, every point and, per physical
//! camera (sensor), a chosen subset of its calibration - Triggs et al.,
//! "Bundle Adjustment - A Modern Synthesis" (1999), §4 and §6.
//!
//! Projection and its Jacobians in the direction and in the calibration come
//! from `camera::Intrinsics` (`project_param_jac`), so every lens model the
//! workspace knows - pinhole, Brown-Conrady, Kannala-Brandt fisheye - is
//! adjusted by the same code. Which calibration parameters move is a list of
//! [`Param`]s per sensor; `fx = fy` tied is the one focal [`Param::Focal`].
//!
//! The points are eliminated with the Schur complement, so the system solved
//! is the dense reduced CAMERA system (pose and calibration columns); each
//! point is then recovered from its own 3x3 block. Residuals pass through a
//! Huber loss (IRLS weights) so a mismatch that survived verification pulls
//! with bounded force. Soft Gaussian priors hold a focal length near a
//! metadata value and the principal point near the image centre, weak enough
//! that only evidence moves them and strong enough that no evidence leaves
//! them where they were.
//!
//! THE GAUGE is explicit - photographs fix a reconstruction only up to a
//! similarity, and the seven free directions are removed rather than left to
//! the damping:
//!
//! * the ANCHOR camera's pose is not a parameter (rotation and translation,
//!   six directions);
//! * the SCALE camera's centre moves only perpendicular to the line from the
//!   anchor's centre (one direction), and after every accepted step the
//!   whole reconstruction is rescaled about the anchor's centre so that the
//!   distance between the two centres is EXACTLY what it was - the tangent
//!   step keeps it to first order, the rescale (a similarity, which changes
//!   no reprojection) keeps it to rounding.

use crate::camera::Pose;
use crate::linalg::{add, cholesky_solve, cross, exp_so3, inv3_spd, mm, mv, norm, normalize, scale, sub, M3, V3};
use ::camera::Intrinsics;

/// One image measurement of one point.
#[derive(Clone, Copy, Debug)]
pub struct Observation {
    pub cam: usize,
    pub point: usize,
    pub px: [f64; 2],
}

/// One calibration degree of freedom bundle adjustment can refine, over
/// `camera::Intrinsics::params` (`fx, fy, cx, cy`, then the lens
/// coefficients).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Param {
    /// `fx` and `fy` moved together: square pixels.
    Focal,
    Fx,
    Fy,
    Cx,
    Cy,
    /// Lens coefficient `i` of `camera::Lens::coeffs` (Brown: `k1..k6` are
    /// 0..5, `p1, p2` are 6, 7; fisheye: `k1..k4` are 0..3).
    Coeff(usize),
}

impl Param {
    /// This parameter's column of the projection's calibration Jacobian.
    fn column(self, jp: &[[f64; 2]]) -> [f64; 2] {
        match self {
            Param::Focal => [jp[0][0] + jp[1][0], jp[0][1] + jp[1][1]],
            Param::Fx => jp[0],
            Param::Fy => jp[1],
            Param::Cx => jp[2],
            Param::Cy => jp[3],
            Param::Coeff(i) => jp[4 + i],
        }
    }

    fn apply(self, p: &mut [f64], d: f64) {
        match self {
            Param::Focal => {
                p[0] += d;
                p[1] += d;
            }
            Param::Fx => p[0] += d,
            Param::Fy => p[1] += d,
            Param::Cx => p[2] += d,
            Param::Cy => p[3] += d,
            Param::Coeff(i) => p[4 + i] += d,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BaCfg {
    pub iters: usize,
    /// Huber threshold in pixels.
    pub huber: f64,
    /// Per sensor, the calibration parameters refined; a sensor with no
    /// entry (or an empty one) keeps its calibration.
    pub free: Vec<Vec<Param>>,
    /// The camera whose pose is held: the gauge's rotation and translation.
    pub anchor: Option<usize>,
    /// The camera whose distance from the anchor is held: the gauge's scale.
    /// Needs `anchor`.
    pub scale: Option<usize>,
    /// Per sensor, a soft prior `(focal px, sigma px)` on a free focal
    /// length (EXIF), `None` for none.
    pub focal_prior: Vec<Option<(f64, f64)>>,
    /// Sigma of the prior pulling a free principal point toward the image
    /// centre, as a fraction of the longer image side.
    pub pp_prior: f64,
}

impl Default for BaCfg {
    fn default() -> Self {
        BaCfg { iters: 50, huber: 2.0, free: Vec::new(), anchor: Some(0), scale: None, focal_prior: Vec::new(), pp_prior: 0.02 }
    }
}

/// Reprojection statistics before and after.
#[derive(Clone, Copy, Debug)]
pub struct BaReport {
    pub rms_before: f64,
    pub rms_after: f64,
    /// Huber cost of the reprojections after, priors excluded.
    pub cost_after: f64,
    pub iterations: usize,
}

/// Residual and Jacobians of one observation.
struct Lin {
    r: [f64; 2],
    /// d r / d (camera point)
    jd: [[f64; 3]; 2],
    /// d r / d (every calibration parameter), empty unless asked for
    jk: Vec<[f64; 2]>,
    /// the camera-frame point
    c: V3,
}

fn linearize(k: &Intrinsics, p: &Pose, x: V3, obs: [f64; 2], want_k: bool) -> Option<Lin> {
    let c = p.to_cam(x);
    let (uv, jd, jk) = if want_k {
        k.project_param_jac(c)?
    } else {
        let (uv, jd) = k.project_jac(c)?;
        (uv, jd, Vec::new())
    };
    Some(Lin { r: [uv[0] - obs[0], uv[1] - obs[1]], jd, jk, c })
}

fn huber_weight(r: &[f64; 2], delta: f64) -> f64 {
    let n = (r[0] * r[0] + r[1] * r[1]).sqrt();
    if n <= delta { 1.0 } else { delta / n }
}

fn huber(r: &[f64; 2], delta: f64) -> f64 {
    let n2 = r[0] * r[0] + r[1] * r[1];
    let n = n2.sqrt();
    if n <= delta { n2 } else { 2.0 * delta * n - delta * delta }
}

/// What an observation the lens cannot image costs: that of a residual of
/// ten Huber thresholds, so a step that pushes points out of the field is
/// refused rather than rewarded for dropping their error.
fn lost(delta: f64) -> f64 {
    huber(&[10.0 * delta, 0.0], delta)
}

/// Huber cost of every observation's reprojection, pixels².
pub fn huber_cost(ks: &[Intrinsics], sensor: &[usize], poses: &[Pose], pts: &[V3], obs: &[Observation], delta: f64) -> f64 {
    obs.iter()
        .map(|o| match crate::camera::project(&ks[sensor[o.cam]], &poses[o.cam], pts[o.point]) {
            Some(uv) => huber(&[uv[0] - o.px[0], uv[1] - o.px[1]], delta),
            None => lost(delta),
        })
        .sum()
}

/// RMS pixel reprojection error over `obs` (observations the lens cannot
/// image skipped).
pub fn rms(ks: &[Intrinsics], sensor: &[usize], poses: &[Pose], pts: &[V3], obs: &[Observation]) -> f64 {
    let (mut s, mut n) = (0.0f64, 0usize);
    for o in obs {
        if let Some(uv) = crate::camera::project(&ks[sensor[o.cam]], &poses[o.cam], pts[o.point]) {
            s += (uv[0] - o.px[0]).powi(2) + (uv[1] - o.px[1]).powi(2);
            n += 1;
        }
    }
    (s / n.max(1) as f64).sqrt()
}

/// One prior residual: `(value - target) / sigma` on the parameter in
/// column `col`, whose value moves one-for-one with it.
struct Prior {
    col: usize,
    sensor: usize,
    /// index into `Intrinsics::params`
    param: usize,
    target: f64,
    sigma: f64,
}

impl Prior {
    fn residual(&self, ks: &[Intrinsics]) -> f64 {
        (ks[self.sensor].params()[self.param] - self.target) / self.sigma
    }
}

fn prior_cost(priors: &[Prior], ks: &[Intrinsics]) -> f64 {
    priors.iter().map(|p| p.residual(ks).powi(2)).sum()
}

/// A camera's place in the reduced system.
#[derive(Clone, Copy)]
enum Slot {
    Held,
    /// rotation increment and translation, six columns from here
    Free(usize),
    /// rotation increment and a centre step across the baseline, five
    Scale(usize),
}

/// Refine the calibrations `ks` (one per sensor; `sensor[c]` is camera
/// `c`'s), `poses` and `points` in place against `obs`.
pub fn bundle_adjust(ks: &mut [Intrinsics], sensor: &[usize], poses: &mut [Pose], points: &mut [V3], obs: &[Observation], cfg: &BaCfg) -> BaReport {
    assert_eq!(sensor.len(), poses.len(), "one sensor per camera");
    assert!(cfg.scale.is_none() || cfg.anchor.is_some(), "a scale camera needs an anchor");
    assert!(cfg.scale != cfg.anchor || cfg.scale.is_none(), "the scale camera cannot be the anchor");
    let nc = poses.len();
    let mut slot = vec![Slot::Held; nc];
    let mut c = 0usize;
    for (i, s) in slot.iter_mut().enumerate() {
        if Some(i) == cfg.anchor {
            continue;
        }
        if Some(i) == cfg.scale {
            *s = Slot::Scale(c);
            c += 5;
        } else {
            *s = Slot::Free(c);
            c += 6;
        }
    }
    let free: Vec<&[Param]> = (0..ks.len()).map(|s| cfg.free.get(s).map_or(&[][..], |v| v.as_slice())).collect();
    let mut kofs = vec![0usize; ks.len()];
    let mut priors = Vec::new();
    for s in 0..ks.len() {
        kofs[s] = c;
        let side = ks[s].width.max(ks[s].height) as f64;
        for (j, p) in free[s].iter().enumerate() {
            let col = c + j;
            match p {
                Param::Focal | Param::Fx | Param::Fy => {
                    if let Some(Some((f0, sigma))) = cfg.focal_prior.get(s) {
                        let param = if *p == Param::Fy { 1 } else { 0 };
                        priors.push(Prior { col, sensor: s, param, target: *f0, sigma: *sigma });
                    }
                }
                Param::Cx if cfg.pp_prior > 0.0 => priors.push(Prior { col, sensor: s, param: 2, target: ks[s].width as f64 / 2.0, sigma: cfg.pp_prior * side }),
                Param::Cy if cfg.pp_prior > 0.0 => priors.push(Prior { col, sensor: s, param: 3, target: ks[s].height as f64 / 2.0, sigma: cfg.pp_prior * side }),
                _ => {}
            }
        }
        c += free[s].len();
    }
    // distance the gauge holds
    let span = match (cfg.anchor, cfg.scale) {
        (Some(a), Some(s)) => Some(norm(sub(poses[s].centre(), poses[a].centre()))),
        _ => None,
    };
    let mut by_point: Vec<Vec<usize>> = vec![Vec::new(); points.len()];
    for (i, o) in obs.iter().enumerate() {
        by_point[o.point].push(i);
    }
    let rms_before = rms(ks, sensor, poses, points, obs);
    let mut lambda = 1e-3;
    let total = |ks: &[Intrinsics], poses: &[Pose], points: &[V3]| huber_cost(ks, sensor, poses, points, obs, cfg.huber) + prior_cost(&priors, ks);
    let mut cost = total(ks, poses, points);
    let mut it = 0;
    while it < cfg.iters {
        it += 1;
        // the scale camera's centre moves across its baseline only
        let across: Option<(V3, V3)> = match (cfg.anchor, cfg.scale) {
            (Some(a), Some(s)) => {
                let b = normalize(sub(poses[s].centre(), poses[a].centre()));
                Some(crate::twoview::tangent_basis(b))
            }
            _ => None,
        };
        // ---- normal equations ----
        let mut u = vec![0.0f64; c * c];
        let mut gc = vec![0.0f64; c];
        let mut vblocks = vec![[0.0f64; 9]; points.len()];
        let mut gp = vec![[0.0f64; 3]; points.len()];
        // per observation: local camera-side columns and W = Jcᵀ Jp (cols x 3)
        let mut local: Vec<(Vec<usize>, Vec<[f64; 3]>)> = Vec::with_capacity(obs.len());
        for o in obs {
            let s = sensor[o.cam];
            let pose = &poses[o.cam];
            let Some(l) = linearize(&ks[s], pose, points[o.point], o.px, !free[s].is_empty()) else {
                local.push((Vec::new(), Vec::new()));
                continue;
            };
            let w = huber_weight(&l.r, cfg.huber);
            let jd = l.jd;
            let chain = |d: V3| -> [f64; 2] { [jd[0][0] * d[0] + jd[0][1] * d[1] + jd[0][2] * d[2], jd[1][0] * d[0] + jd[1][1] * d[1] + jd[1][2] * d[2]] };
            let mut cols = Vec::with_capacity(6 + free[s].len());
            let mut jrows: Vec<[f64; 2]> = Vec::with_capacity(6 + free[s].len());
            // left rotation increment ω: the camera point turns about the
            // camera centre, d c / d ω = −[c − t]ₓ with t held, −[c]ₓ with the
            // centre held
            let about = |held_centre: bool| if held_centre { l.c } else { sub(l.c, pose.t) };
            match slot[o.cam] {
                Slot::Held => {}
                Slot::Free(at) => {
                    let q = about(false);
                    for (j, e) in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]].into_iter().enumerate() {
                        cols.push(at + j);
                        jrows.push(chain(cross(e, q)));
                    }
                    for (j, e) in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]].into_iter().enumerate() {
                        cols.push(at + 3 + j);
                        jrows.push(chain(e));
                    }
                }
                Slot::Scale(at) => {
                    let q = about(true);
                    for (j, e) in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]].into_iter().enumerate() {
                        cols.push(at + j);
                        jrows.push(chain(cross(e, q)));
                    }
                    // c = R (X − C): d c / d C = −R
                    let (b1, b2) = across.unwrap();
                    for (j, b) in [b1, b2].into_iter().enumerate() {
                        cols.push(at + 3 + j);
                        jrows.push(chain(scale(mv(&pose.r, b), -1.0)));
                    }
                }
            }
            for (j, p) in free[s].iter().enumerate() {
                cols.push(kofs[s] + j);
                jrows.push(p.column(&l.jk));
            }
            // d c / d X = R
            let jp: [[f64; 3]; 2] = std::array::from_fn(|r| std::array::from_fn(|k| (0..3).map(|m| jd[r][m] * pose.r[m * 3 + k]).sum()));
            for (a, ja) in cols.iter().zip(&jrows) {
                gc[*a] += w * (ja[0] * l.r[0] + ja[1] * l.r[1]);
                for (b, jb) in cols.iter().zip(&jrows) {
                    u[a * c + b] += w * (ja[0] * jb[0] + ja[1] * jb[1]);
                }
            }
            let vb = &mut vblocks[o.point];
            for a in 0..3 {
                gp[o.point][a] += w * (jp[0][a] * l.r[0] + jp[1][a] * l.r[1]);
                for b in 0..3 {
                    vb[a * 3 + b] += w * (jp[0][a] * jp[0][b] + jp[1][a] * jp[1][b]);
                }
            }
            let wm: Vec<[f64; 3]> = jrows.iter().map(|ja| std::array::from_fn(|b| w * (ja[0] * jp[0][b] + ja[1] * jp[1][b]))).collect();
            local.push((cols, wm));
        }
        for p in &priors {
            let r = p.residual(ks);
            gc[p.col] += r / p.sigma;
            u[p.col * c + p.col] += 1.0 / (p.sigma * p.sigma);
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
                let vbp = mv(&vi, bp);
                for &oa in list {
                    let (ca, wa) = &local[oa];
                    for (ia, wr) in ca.iter().zip(wa) {
                        rhs[*ia] -= wr[0] * vbp[0] + wr[1] * vbp[1] + wr[2] * vbp[2];
                    }
                    for &ob in list {
                        let (cb, wb) = &local[ob];
                        for (ia, wra) in ca.iter().zip(wa) {
                            let t = mv(&vi, *wra);
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
                np[p] = add(points[p], mv(&vinv[p], b));
            }
            let mut npose = poses.to_vec();
            for (i, sl) in slot.iter().enumerate() {
                match *sl {
                    Slot::Held => {}
                    Slot::Free(at) => {
                        let r: M3 = mm(&exp_so3([dc[at], dc[at + 1], dc[at + 2]]), &poses[i].r);
                        npose[i] = Pose { r, t: add(poses[i].t, [dc[at + 3], dc[at + 4], dc[at + 5]]) };
                    }
                    Slot::Scale(at) => {
                        let r: M3 = mm(&exp_so3([dc[at], dc[at + 1], dc[at + 2]]), &poses[i].r);
                        let (b1, b2) = across.unwrap();
                        let centre = add(poses[i].centre(), add(scale(b1, dc[at + 3]), scale(b2, dc[at + 4])));
                        npose[i] = Pose { r, t: scale(mv(&r, centre), -1.0) };
                    }
                }
            }
            let mut nk = ks.to_vec();
            for s in 0..ks.len() {
                if free[s].is_empty() {
                    continue;
                }
                let mut p = ks[s].params();
                for (j, prm) in free[s].iter().enumerate() {
                    prm.apply(&mut p, dc[kofs[s] + j]);
                }
                nk[s] = ks[s].with_params(&p);
            }
            if let (Some(a), Some(sc), Some(d0)) = (cfg.anchor, cfg.scale, span) {
                hold_scale(&mut npose, &mut np, a, sc, d0);
            }
            let ncost = total(&nk, &npose, &np);
            if ncost < cost && nk.iter().all(|k| k.fx > 0.0 && k.fy > 0.0) {
                let gain = cost - ncost;
                ks.copy_from_slice(&nk);
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
    BaReport {
        rms_before,
        rms_after: rms(ks, sensor, poses, points, obs),
        cost_after: huber_cost(ks, sensor, poses, points, obs, cfg.huber),
        iterations: it,
    }
}

/// Rescale everything but the anchor about the anchor's centre so the scale
/// camera sits exactly `d0` from it: a similarity, invisible to every
/// reprojection.
fn hold_scale(poses: &mut [Pose], points: &mut [V3], anchor: usize, sc: usize, d0: f64) {
    let ca = poses[anchor].centre();
    let d = norm(sub(poses[sc].centre(), ca));
    if d <= 0.0 || !d.is_finite() {
        return;
    }
    let s = d0 / d;
    for (i, p) in poses.iter_mut().enumerate() {
        if i != anchor {
            let c = add(ca, scale(sub(p.centre(), ca), s));
            p.t = scale(mv(&p.r, c), -1.0);
        }
    }
    for x in points.iter_mut() {
        *x = add(ca, scale(sub(*x, ca), s));
    }
}

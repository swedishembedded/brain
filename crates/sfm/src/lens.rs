// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which lens model describes a camera: every candidate is fitted to the
//! same reconstruction by bundle adjustment and the Bayesian information
//! criterion (Schwarz, "Estimating the Dimension of a Model", Annals of
//! Statistics 1978) picks between them,
//!
//! `BIC = n ln(C / n) + k ln n`,
//!
//! with `n` the number of residuals (two per observation), `C` their robust
//! (Huber) cost and `k` the number of calibration parameters - the poses and
//! points are common to every candidate and cancel. A coefficient has to
//! lower the cost by a factor `n^(1/n)` per parameter to be worth having, so
//! one that only fits noise loses and a model that cannot follow the lens
//! loses by its residuals.
//!
//! The candidates:
//!
//! * [`LensChoice::Pinhole`] - `fx`, `cx`, `cy`;
//! * [`LensChoice::Radial`] - Brown-Conrady with `k1, k2`;
//! * [`LensChoice::Brown`] - Brown-Conrady with `k1, k2, k3, p1, p2`
//!   (Brown 1966, as OpenCV parameterizes it);
//! * [`LensChoice::Fisheye`] - Kannala-Brandt `k1..k4` (TPAMI 2006).
//!
//! A candidate starts from the reconstruction's own calibration CONVERTED to
//! its model: the source lens's radial map - distorted radius as a function
//! of the angle off axis - is sampled over the angles the observations
//! actually cover and the candidate's radial polynomial is fitted to it by
//! least squares, so a fisheye starts where the radial camera was, not at an
//! equidistant guess.
//!
//! Swedish Embedded AB implements camera calibration and lens modelling for
//! photogrammetry and embedded vision. If your team needs cameras calibrated
//! from ordinary photographs, you can procure our services by sending an
//! email to info@swedishembedded.com.

use crate::ba::{bundle_adjust, BaCfg, Observation, Param};
use crate::camera::Pose;
use crate::linalg::{cholesky_solve, V3};
use ::camera::{Intrinsics, Lens};

/// A lens model, or `Auto` to choose among all of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LensChoice {
    Auto,
    Pinhole,
    Radial,
    Brown,
    Fisheye,
}

/// How much of a camera's calibration bundle adjustment may move, by how
/// well the reconstruction constrains it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    /// Too few views: the calibration is held.
    Held,
    /// The focal length and the two lowest lens coefficients.
    Focal,
    /// Everything the model has, principal point included.
    Full,
}

/// The widest angle off axis a Brown-Conrady fit is sampled to: the model's
/// radius `tan θ` runs away past it, and a lens that needs more is a fisheye.
const BROWN_MAX_ANGLE: f64 = 75.0 * std::f64::consts::PI / 180.0;

impl LensChoice {
    /// The models `self` stands for.
    pub fn candidates(self) -> Vec<LensChoice> {
        match self {
            LensChoice::Auto => vec![LensChoice::Pinhole, LensChoice::Radial, LensChoice::Brown, LensChoice::Fisheye],
            one => vec![one],
        }
    }

    /// The model a calibration is written in.
    pub fn of(lens: &Lens) -> LensChoice {
        match lens {
            Lens::Pinhole | Lens::Equirect => LensChoice::Pinhole,
            Lens::Brown { k, p, s } => {
                if k[2..].iter().chain(p).chain(s).all(|v| *v == 0.0) {
                    LensChoice::Radial
                } else {
                    LensChoice::Brown
                }
            }
            Lens::Fisheye { .. } => LensChoice::Fisheye,
        }
    }

    /// The lens coefficients (indices into `Lens::coeffs`) the model refines.
    fn coeffs(self) -> &'static [usize] {
        match self {
            LensChoice::Auto | LensChoice::Pinhole => &[],
            LensChoice::Radial => &[0, 1],
            LensChoice::Brown => &[0, 1, 2, 6, 7],
            LensChoice::Fisheye => &[0, 1, 2, 3],
        }
    }

    /// The calibration parameters bundle adjustment refines at `stage`;
    /// `square` ties `fx = fy`.
    pub fn params(self, stage: Stage, square: bool) -> Vec<Param> {
        let mut v = Vec::new();
        if stage == Stage::Held {
            return v;
        }
        if square {
            v.push(Param::Focal);
        } else {
            v.extend([Param::Fx, Param::Fy]);
        }
        let coeffs = self.coeffs();
        match stage {
            Stage::Focal => v.extend(coeffs.iter().take(2).map(|&i| Param::Coeff(i))),
            _ => {
                v.extend([Param::Cx, Param::Cy]);
                v.extend(coeffs.iter().map(|&i| Param::Coeff(i)));
            }
        }
        v
    }

    /// `k` rewritten in this model: the linear part kept, the lens fitted to
    /// `k`'s radial map over angles off axis up to `theta_max` radians.
    pub fn convert(self, k: &Intrinsics, theta_max: f64) -> Intrinsics {
        let from = LensChoice::of(&k.lens);
        let lens = match (self, from) {
            (LensChoice::Auto, _) => k.lens,
            (a, b) if a == b => k.lens,
            (LensChoice::Pinhole, _) => Lens::Pinhole,
            (LensChoice::Radial | LensChoice::Brown, LensChoice::Radial | LensChoice::Brown) => {
                // same family: keep the terms both have
                let c = k.lens.coeffs();
                let mut kk = [0.0; 6];
                kk[0] = c[0];
                kk[1] = c[1];
                let (mut p, mut k3) = ([0.0; 2], 0.0);
                if self == LensChoice::Brown {
                    k3 = c[2];
                    p = [c[6], c[7]];
                }
                kk[2] = k3;
                Lens::Brown { k: kk, p, s: [0.0; 4] }
            }
            (LensChoice::Radial | LensChoice::Brown, _) => {
                let terms = if self == LensChoice::Radial { 2 } else { 3 };
                let top = theta_max.min(BROWN_MAX_ANGLE);
                // rd / r − 1 = Σ k_i r^(2i), r = tan θ
                let c = fit_poly(&k.lens, top, terms, |t| t.tan(), |t, i| t.tan().powi(2 * i as i32 + 2), |rd, t| rd / t.tan() - 1.0);
                let mut kk = [0.0; 6];
                kk[..terms].copy_from_slice(&c);
                Lens::Brown { k: kk, p: [0.0; 2], s: [0.0; 4] }
            }
            (LensChoice::Fisheye, _) => {
                // θd − θ = Σ k_i θ^(2i+3)
                let c = fit_poly(&k.lens, theta_max, 4, |_| 1.0, |t, i| t.powi(2 * i as i32 + 3), |rd, t| rd - t);
                Lens::Fisheye { k: [c[0], c[1], c[2], c[3]] }
            }
        };
        Intrinsics { lens, ..*k }
    }
}

/// Distorted normalized radius of the ray `theta` radians off axis through
/// `lens` - the radial part of the model, `None` where it does not image.
fn radial_map(lens: &Lens, theta: f64) -> Option<f64> {
    match lens {
        Lens::Pinhole => (theta < std::f64::consts::FRAC_PI_2).then(|| theta.tan()),
        Lens::Brown { k, .. } => {
            if theta >= std::f64::consts::FRAC_PI_2 {
                return None;
            }
            let r = theta.tan();
            let r2 = r * r;
            let num = 1.0 + r2 * (k[0] + r2 * (k[1] + r2 * k[2]));
            let den = 1.0 + r2 * (k[3] + r2 * (k[4] + r2 * k[5]));
            Some(r * num / den)
        }
        Lens::Fisheye { k } => {
            let t2 = theta * theta;
            Some(theta * (1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3])))))
        }
        Lens::Equirect => None,
    }
}

/// Least-squares coefficients `c` of `target(rd, θ) ≈ Σ c_i basis(θ, i)`
/// over the angles in `(0, top]` where `source` images, each residual
/// multiplied by `scale(θ)` - the distorted radius per unit of target - so
/// the fit is uniform in image distance.
fn fit_poly(source: &Lens, top: f64, terms: usize, scale: impl Fn(f64) -> f64, basis: impl Fn(f64, usize) -> f64, target: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    let samples = 64;
    let mut ata = vec![0.0f64; terms * terms];
    let mut atb = vec![0.0f64; terms];
    for s in 1..=samples {
        let t = top * s as f64 / samples as f64;
        let Some(rd) = radial_map(source, t) else { continue };
        let w = scale(t);
        let y = target(rd, t) * w;
        let row: Vec<f64> = (0..terms).map(|i| basis(t, i) * w).collect();
        for i in 0..terms {
            atb[i] += row[i] * y;
            for j in 0..terms {
                ata[i * terms + j] += row[i] * row[j];
            }
        }
    }
    let ridge = 1e-12 * (0..terms).map(|i| ata[i * terms + i]).fold(0.0, f64::max).max(1e-300);
    for i in 0..terms {
        ata[i * terms + i] += ridge;
    }
    cholesky_solve(&ata, &atb, terms).unwrap_or_else(|| vec![0.0; terms])
}

/// Model selection settings.
#[derive(Clone, Debug)]
pub struct LensCfg {
    pub choice: LensChoice,
    /// Tie `fx = fy`.
    pub square_pixels: bool,
    /// Views a sensor needs before its full model (principal point and
    /// every coefficient) is refined.
    pub full_model_views: usize,
    /// The adjustment every candidate runs; its `free` is replaced per
    /// candidate.
    pub ba: BaCfg,
}

impl Default for LensCfg {
    fn default() -> Self {
        LensCfg { choice: LensChoice::Auto, square_pixels: true, full_model_views: 6, ba: BaCfg { iters: 100, ..BaCfg::default() } }
    }
}

/// How one candidate model fitted.
#[derive(Clone, Debug)]
pub struct LensFit {
    pub choice: LensChoice,
    /// The refined calibration, one per sensor.
    pub intrinsics: Vec<Intrinsics>,
    pub rms_px: f64,
    /// Huber cost of the reprojections, pixels².
    pub cost: f64,
    /// Calibration parameters refined.
    pub params: usize,
    pub bic: f64,
}

/// A candidate model fitted, with the poses and points it adjusted to.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub fit: LensFit,
    pub poses: Vec<Pose>,
    pub points: Vec<V3>,
}

/// How far a sensor's calibration may move given how many views it has.
pub fn stage_for(views: usize, full_model_views: usize) -> Stage {
    if views >= full_model_views {
        Stage::Full
    } else if views >= 3 {
        Stage::Focal
    } else {
        Stage::Held
    }
}

/// Fit every candidate model of `cfg.choice` to the reconstruction (`ks`
/// per sensor, `sensor[c]` of camera `c`). The candidate in the model `ks`
/// is already written in is refined first and every other one is converted
/// from its result. Candidates come back in [`LensChoice::candidates`]
/// order; the one with the lowest `fit.bic` is the choice.
pub fn fit_lenses(ks: &[Intrinsics], sensor: &[usize], poses: &[Pose], points: &[V3], obs: &[Observation], cfg: &LensCfg) -> Vec<Candidate> {
    let nsensors = ks.len();
    let mut views = vec![std::collections::BTreeSet::new(); nsensors];
    let mut theta_max = vec![0.0f64; nsensors];
    for o in obs {
        let s = sensor[o.cam];
        views[s].insert(o.cam);
        if let Some(d) = ks[s].unproject(o.px) {
            theta_max[s] = theta_max[s].max(d[2].clamp(-1.0, 1.0).acos());
        }
    }
    let stages: Vec<Stage> = views.iter().map(|v| stage_for(v.len(), cfg.full_model_views)).collect();
    let n = 2.0 * obs.len() as f64;
    let run = |choice: LensChoice, start: &[Intrinsics], poses: &[Pose], points: &[V3]| -> Candidate {
        let mut k: Vec<Intrinsics> = start.iter().zip(&theta_max).map(|(k, t)| choice.convert(k, *t)).collect();
        let (mut p, mut x) = (poses.to_vec(), points.to_vec());
        // the low-order terms first, from where the conversion left them,
        // then everything the sensor's views support
        let low: Vec<Vec<Param>> = stages.iter().map(|s| choice.params((*s).min(Stage::Focal), cfg.square_pixels)).collect();
        bundle_adjust(&mut k, sensor, &mut p, &mut x, obs, &BaCfg { free: low, iters: 30, ..cfg.ba.clone() });
        let free: Vec<Vec<Param>> = stages.iter().map(|s| choice.params(*s, cfg.square_pixels)).collect();
        let params = free.iter().map(|f| f.len()).sum::<usize>();
        let rep = bundle_adjust(&mut k, sensor, &mut p, &mut x, obs, &BaCfg { free, ..cfg.ba.clone() });
        let bic = n * (rep.cost_after.max(1e-300) / n).ln() + params as f64 * n.ln();
        Candidate { fit: LensFit { choice, intrinsics: k, rms_px: rep.rms_after, cost: rep.cost_after, params, bic }, poses: p, points: x }
    };
    let base_choice = ks.first().map_or(LensChoice::Radial, |k| LensChoice::of(&k.lens));
    let base = run(base_choice, ks, poses, points);
    let wanted = cfg.choice.candidates();
    let others: Vec<LensChoice> = wanted.iter().copied().filter(|c| *c != base_choice).collect();
    let fitted = backend_cpu::par::map(others.len(), |i| run(others[i], &base.fit.intrinsics, &base.poses, &base.points));
    let mut out: Vec<Candidate> = Vec::with_capacity(wanted.len());
    let mut fitted = fitted.into_iter();
    let mut base = Some(base);
    for c in wanted {
        if c == base_choice {
            out.push(base.take().unwrap());
        } else {
            out.push(fitted.next().unwrap());
        }
    }
    out
}

/// One line describing a calibration for a log: focal length(s), principal
/// point, model and its non-zero coefficients.
pub fn describe(k: &Intrinsics) -> String {
    let focal = if k.fx == k.fy { format!("f {:.2}", k.fx) } else { format!("fx {:.2} fy {:.2}", k.fx, k.fy) };
    let mut s = format!("{focal} c ({:.2}, {:.2}) {}", k.cx, k.cy, k.lens.name());
    for (name, v) in k.param_names()[4..].iter().zip(k.lens.coeffs()) {
        if v != 0.0 {
            s += &format!(" {name} {v:+.5}");
        }
    }
    s
}

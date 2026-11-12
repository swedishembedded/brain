// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The rigid head fit: 2D landmarks through a calibrated lens to a 6-DoF head
//! pose and the two eyes in the camera frame.
//!
//! Landmark pixels become unit bearings through `camera::Intrinsics` (so lens
//! distortion is handled by the one lens model, not approximated here), and
//! `sfm::pnp::refine` minimises the tangent-plane reprojection error of the
//! canonical face against them. Translation and rotation are solved together,
//! which is what separates "the head turned" from "the head moved": an
//! inter-eye distance that shrinks because of yaw does not read as the viewer
//! backing away.
//!
//! Seeding: a tracked head starts from the previous frame's pose; a new one
//! starts facing the camera at the depth its outer eye corners imply. Faces a
//! landmark model returns are within about +-60 degrees of frontal, which is
//! well inside where the frontal seed converges.

use camera::Intrinsics;
use sfm::camera::Pose;
use sfm::linalg::{dot, norm, sub, I3};

use crate::face::FaceModel;
use crate::V3;

/// A fitted head.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeadFit {
    /// Head frame to camera frame: `x_cam = R x_head + t`.
    pub head_to_camera: Pose,
    /// The subject's right pupil in the camera frame.
    pub right_eye: V3,
    /// The subject's left pupil in the camera frame.
    pub left_eye: V3,
    /// Root-mean-square reprojection error over the landmarks used, pixels.
    pub rmse_px: f64,
    /// How many landmarks entered the fit.
    pub used: usize,
}

impl HeadFit {
    /// The midpoint of the pupils, camera frame - the head frame's origin.
    pub fn cyclopean_eye(&self) -> V3 {
        self.head_to_camera.t
    }
}

/// Why a fit was refused. Every variant is a measurement the estimator should
/// drop, not a pose it should render.
#[derive(Clone, Debug, PartialEq)]
pub enum FitError {
    /// Fewer landmarks than the model indexes, or non-finite coordinates.
    BadLandmarks(String),
    /// Too few confident rigid landmarks to constrain six degrees of freedom.
    TooFewPoints { usable: usize, needed: usize },
    /// The solver converged to something that is not a face in front of the
    /// camera, or that does not explain the landmarks.
    Implausible(String),
}

impl std::fmt::Display for FitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FitError::BadLandmarks(why) => write!(f, "landmarks refused: {why}"),
            FitError::TooFewPoints { usable, needed } => write!(f, "{usable} confident rigid landmarks, a head fit needs {needed}"),
            FitError::Implausible(why) => write!(f, "head fit rejected: {why}"),
        }
    }
}

impl std::error::Error for FitError {}

/// The fewest rigid points a fit accepts. Six DoF need three, but with fewer
/// than six a single mislocated landmark decides the pose.
pub const MIN_POINTS: usize = 6;

/// Solver iterations; the refine stops early when a step stops helping.
const ITERS: usize = 30;

/// Accept range for the head's distance from the camera, metres. Outside it
/// the fit is a misdetection, not a viewer.
const DEPTH_RANGE_M: (f64, f64) = (0.1, 3.0);

/// Fit `model` to `points_px` (one per landmark index, in pixels of the image
/// `intrinsics` describes), using only landmarks whose `confidence` is at
/// least `min_confidence`. `prior` is the previous frame's pose, if the same
/// head was tracked there. `max_rmse_px` bounds the accepted residual.
pub fn fit_head(
    points_px: &[[f64; 2]],
    confidence: &[f64],
    intrinsics: &Intrinsics,
    model: &FaceModel,
    prior: Option<&Pose>,
    min_confidence: f64,
    max_rmse_px: f64,
) -> Result<HeadFit, FitError> {
    if points_px.len() <= model.max_index() || confidence.len() != points_px.len() {
        return Err(FitError::BadLandmarks(format!(
            "{} points and {} confidences; the face model reads index {}",
            points_px.len(),
            confidence.len(),
            model.max_index()
        )));
    }
    let mut x = Vec::with_capacity(model.points.len());
    let mut f = Vec::with_capacity(model.points.len());
    let mut px = Vec::with_capacity(model.points.len());
    for &(i, p) in &model.points {
        let q = points_px[i];
        if !(q[0].is_finite() && q[1].is_finite()) {
            return Err(FitError::BadLandmarks(format!("landmark {i} is {q:?}")));
        }
        if confidence[i].is_nan() || confidence[i] < min_confidence {
            continue;
        }
        if let Some(b) = intrinsics.unproject(q) {
            x.push(p);
            f.push(b);
            px.push(q);
        }
    }
    if x.len() < MIN_POINTS {
        return Err(FitError::TooFewPoints { usable: x.len(), needed: MIN_POINTS });
    }

    let seed = match prior {
        Some(p) => *p,
        None => frontal_seed(points_px, intrinsics, model)?,
    };
    let pose = sfm::pnp::refine(seed, &x, &f, ITERS);

    let t = pose.t;
    if !(t.iter().all(|c| c.is_finite()) && t[2] > DEPTH_RANGE_M.0 && t[2] < DEPTH_RANGE_M.1) {
        return Err(FitError::Implausible(format!("head at {t:?} m from the camera")));
    }
    // the face must point at the camera: the head's +Z (into the head) has to
    // point away from the lens
    let facing = pose.r[8];
    if facing <= 0.2 {
        return Err(FitError::Implausible(format!("head turned away from the camera (axis cosine {facing:.2})")));
    }
    let mut se = 0.0;
    for (p, q) in x.iter().zip(&px) {
        let Some(r) = intrinsics.project(pose.to_cam(*p)) else {
            return Err(FitError::Implausible("a fitted landmark projects outside the lens model".into()));
        };
        se += (r[0] - q[0]).powi(2) + (r[1] - q[1]).powi(2);
    }
    let rmse_px = (se / x.len() as f64).sqrt();
    if rmse_px.is_nan() || rmse_px > max_rmse_px {
        return Err(FitError::Implausible(format!("reprojection error {rmse_px:.1} px exceeds {max_rmse_px:.1} px")));
    }
    Ok(HeadFit { head_to_camera: pose, right_eye: pose.to_cam(model.right_eye), left_eye: pose.to_cam(model.left_eye), rmse_px, used: x.len() })
}

/// A head facing the camera, at the depth where its outer eye corners would
/// be as far apart on the sensor as the landmarks say, centred on their
/// midpoint's ray.
fn frontal_seed(points_px: &[[f64; 2]], intrinsics: &Intrinsics, model: &FaceModel) -> Result<Pose, FitError> {
    let (a, b) = model.outer_corners();
    let width_m = norm(sub(b, a));
    let (pa, pb) = (points_px[36], points_px[45]);
    let (Some(ra), Some(rb)) = (intrinsics.unproject(pa), intrinsics.unproject(pb)) else {
        return Err(FitError::BadLandmarks("outer eye corners fall outside the lens model".into()));
    };
    // angle subtended by the corners -> distance for a fronto-parallel pair
    let angle = dot(ra, rb).clamp(-1.0, 1.0).acos();
    if angle < 1e-4 {
        return Err(FitError::BadLandmarks("outer eye corners coincide".into()));
    }
    let depth = width_m / (2.0 * (angle / 2.0).tan());
    let mid = intrinsics
        .unproject([(pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0])
        .ok_or_else(|| FitError::BadLandmarks("eye midpoint falls outside the lens model".into()))?;
    // the corners sit behind the pupils by their mean model depth
    let corner_z = (a[2] + b[2]) / 2.0;
    let range = (depth - corner_z) / mid[2].max(1e-6);
    Ok(Pose { r: I3, t: [mid[0] * range, mid[1] * range, mid[2] * range] })
}

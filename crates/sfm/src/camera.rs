// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Camera poses, and projection through the workspace's one camera model
//! (`camera::Intrinsics`, which this crate calibrates).
//!
//! Conventions match `splat::types::Camera`: +X right, +Y down, +Z forward,
//! `X_cam = R·X_world + t`.

use crate::linalg::{add, mtv, mv, scale, M3, V3};
use ::camera::Intrinsics;

/// World-to-camera rigid transform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub r: M3,
    pub t: V3,
}

impl Pose {
    pub fn identity() -> Pose {
        Pose { r: crate::linalg::I3, t: [0.0; 3] }
    }

    pub fn to_cam(&self, x: V3) -> V3 {
        add(mv(&self.r, x), self.t)
    }

    /// Camera centre in world coordinates.
    pub fn centre(&self) -> V3 {
        scale(mtv(&self.r, self.t), -1.0)
    }

    /// Camera-to-world as a row-major 4x4, the layout `splat` uses.
    pub fn c2w(&self) -> [f64; 16] {
        let c = self.centre();
        let r = &self.r;
        [r[0], r[3], r[6], c[0], r[1], r[4], r[7], c[1], r[2], r[5], r[8], c[2], 0.0, 0.0, 0.0, 1.0]
    }
}

/// Where world point `x` lands in the image, `None` where the lens does not
/// image it (behind a perspective camera, past the lens's valid field).
pub fn project(k: &Intrinsics, pose: &Pose, x: V3) -> Option<[f64; 2]> {
    k.project(pose.to_cam(x))
}

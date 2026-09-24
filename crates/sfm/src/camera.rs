// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The camera model structure-from-motion estimates: one physical camera
//! shared by every photograph of a capture (a single focal length and a
//! two-coefficient radial distortion about the image centre), and a rigid
//! pose per photograph.
//!
//! Conventions match `splat::types::Camera`: +X right, +Y down, +Z forward,
//! `X_cam = R·X_world + t`.

use crate::linalg::{add, mtv, mv, scale, M3, V3};

/// Intrinsics shared by every view.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Intrinsics {
    /// Focal length in pixels.
    pub f: f64,
    pub cx: f64,
    pub cy: f64,
    /// Radial distortion `d(r²) = 1 + k1·r² + k2·r⁴` on normalized
    /// coordinates.
    pub k1: f64,
    pub k2: f64,
    pub width: u32,
    pub height: u32,
}

impl Intrinsics {
    /// A first guess for a camera with no metadata: principal point at the
    /// centre, no distortion, and a focal length of `fov_factor` times the
    /// larger side (1.2 is a typical phone main camera, about 45° across the
    /// long side).
    pub fn guess(width: u32, height: u32, fov_factor: f64) -> Intrinsics {
        Intrinsics { f: fov_factor * width.max(height) as f64, cx: width as f64 / 2.0, cy: height as f64 / 2.0, k1: 0.0, k2: 0.0, width, height }
    }

    pub fn distortion(&self, r2: f64) -> f64 {
        1.0 + self.k1 * r2 + self.k2 * r2 * r2
    }

    /// Pixel of a normalized (undistorted) image coordinate.
    pub fn to_pixel(&self, n: [f64; 2]) -> [f64; 2] {
        let d = self.distortion(n[0] * n[0] + n[1] * n[1]);
        [self.f * d * n[0] + self.cx, self.f * d * n[1] + self.cy]
    }

    /// Normalized, undistorted coordinate of a pixel (fixed-point inversion
    /// of the radial model).
    pub fn to_normalized(&self, p: [f64; 2]) -> [f64; 2] {
        let xd = [(p[0] - self.cx) / self.f, (p[1] - self.cy) / self.f];
        let mut x = xd;
        for _ in 0..20 {
            let d = self.distortion(x[0] * x[0] + x[1] * x[1]);
            x = [xd[0] / d, xd[1] / d];
        }
        x
    }
}

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

/// Where world point `x` lands in the image, `None` behind the camera.
pub fn project(k: &Intrinsics, pose: &Pose, x: V3) -> Option<[f64; 2]> {
    let c = pose.to_cam(x);
    if c[2] <= 1e-9 {
        return None;
    }
    Some(k.to_pixel([c[0] / c[2], c[1] / c[2]]))
}

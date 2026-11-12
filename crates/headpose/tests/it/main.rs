// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The head-coupled perspective spec, one module per stage.

mod calibration;
mod estimator;
mod fitting;
mod projection;

/// A pinhole webcam: 640x480, 60 degrees horizontal field of view.
pub fn webcam() -> camera::Intrinsics {
    let f = 320.0 / (30f64.to_radians()).tan();
    camera::Intrinsics::pinhole(f, 640, 480)
}

/// Rotation from yaw (about +Y), pitch (about +X) and roll (about +Z), in
/// degrees, row-major.
pub fn rotation(yaw: f64, pitch: f64, roll: f64) -> [f64; 9] {
    use sfm::linalg::mm;
    let (sy, cy) = yaw.to_radians().sin_cos();
    let (sp, cp) = pitch.to_radians().sin_cos();
    let (sr, cr) = roll.to_radians().sin_cos();
    let ry = [cy, 0.0, sy, 0.0, 1.0, 0.0, -sy, 0.0, cy];
    let rx = [1.0, 0.0, 0.0, 0.0, cp, -sp, 0.0, sp, cp];
    let rz = [cr, -sr, 0.0, sr, cr, 0.0, 0.0, 0.0, 1.0];
    mm(&ry, &mm(&rx, &rz))
}

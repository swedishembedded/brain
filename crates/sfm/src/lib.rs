// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Structure from motion: camera poses, one shared calibration and a sparse
//! point cloud, recovered from nothing but a set of overlapping photographs.

pub mod ba;
pub mod camera;
pub mod incremental;
pub mod linalg;
pub mod matching;
pub mod pnp;
pub mod sift;
pub mod twoview;

/// How many RANSAC samples of `sample` points give 99.9% confidence of
/// drawing at least one all-inlier sample when a fraction `inliers` of the
/// data are inliers (Fischler & Bolles 1981), capped at `cap`.
///
/// A fixed count is either wasted on clean data or too small on dirty data:
/// at 30% inliers a six-point sample is all-inlier once in 1,372 draws, so a
/// fixed 2,000 finds a pose about three times in four.
pub fn ransac_iterations(inliers: f64, sample: i32, cap: usize) -> usize {
    let p = inliers.clamp(1e-9, 1.0).powi(sample);
    if p >= 1.0 - 1e-12 {
        return 1;
    }
    let n = (1.0f64 - 0.999).ln() / (1.0 - p).ln();
    (n.ceil() as usize).clamp(1, cap)
}

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Structure from motion: camera poses, a calibration per physical camera
//! (in the workspace's one camera model, `camera::Intrinsics`, under a
//! selected lens model) and a sparse point cloud, recovered from nothing but
//! a set of overlapping photographs.

pub mod ba;
pub mod camera;
pub mod incremental;
pub mod lens;
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
    let p = inliers.clamp(0.0, 1.0).powi(sample);
    if p >= 1.0 - 1e-12 {
        return 1;
    }
    // ln(1 - p) through ln_1p: for a rare all-inlier sample `1 - p` rounds
    // to exactly 1, and the bound came out as a single iteration
    let n = (1.0f64 - 0.999).ln() / (-p).ln_1p();
    if !n.is_finite() {
        return cap;
    }
    (n.ceil() as usize).clamp(1, cap)
}

#[cfg(test)]
mod tests {
    use super::ransac_iterations;

    /// A model with little or no support must not end the search: the first
    /// sample of a pair can be degenerate, and when it was, the bound used to
    /// come out as ONE iteration and the pair was rejected with its true
    /// matches unexamined.
    #[test]
    fn rare_inliers_keep_the_search_going() {
        assert_eq!(ransac_iterations(0.0, 8, 50_000), 50_000);
        assert_eq!(ransac_iterations(0.05, 8, 50_000), 50_000);
        assert_eq!(ransac_iterations(0.02, 3, 50_000), 50_000);
        // and the textbook value where it is representable: 99.9% at half
        // inliers and eight-point samples is 1765 draws
        assert_eq!(ransac_iterations(0.5, 8, 50_000), 1765);
    }
}

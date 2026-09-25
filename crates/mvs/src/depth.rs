// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A view's dense geometry: per pixel, the range to the surface along the
//! pixel's own unit ray, the surface normal and how far to trust both.
//!
//! This is exactly the prior `splat::opt::TargetView` takes (`depth` =
//! [`DepthMap::range`], `normals` = [`DepthMap::normal`], `depth_conf` =
//! [`DepthMap::conf`]). A fit that runs at a coarser halving than the stereo
//! takes [`DepthMap::halved`], which never averages: a 2x2 block's value is
//! ONE of its samples - the most confident - so a depth edge stays an edge
//! instead of becoming a ramp of points floating between the two surfaces.

use camera::Intrinsics;

/// Dense geometry of one view at `width x height`.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthMap {
    pub width: u32,
    pub height: u32,
    /// `[W*H]` distance from the camera centre along the pixel's unit ray;
    /// 0 = no measurement.
    pub range: Vec<f32>,
    /// `[W*H*3]` unit surface normal in the camera frame (+X right, +Y down,
    /// +Z forward), facing the camera; zero where there is no measurement.
    pub normal: Vec<f32>,
    /// `[W*H]` confidence in [0, 1]; 0 where there is no measurement.
    pub conf: Vec<f32>,
}

impl DepthMap {
    /// A map with no measurements.
    pub fn empty(width: u32, height: u32) -> DepthMap {
        let n = width as usize * height as usize;
        DepthMap { width, height, range: vec![0.0; n], normal: vec![0.0; n * 3], conf: vec![0.0; n] }
    }

    /// Fraction of pixels with a measurement.
    pub fn coverage(&self) -> f64 {
        self.range.iter().filter(|r| **r > 0.0).count() as f64 / self.range.len().max(1) as f64
    }

    /// The map of the same camera imaging half the size (`width / 2 x
    /// height / 2`, rounded down, as `mvs_prepare` halves photographs);
    /// `k` is the calibration at THIS map's size.
    ///
    /// Each output pixel takes the most confident measured sample of its 2x2
    /// block (the first in scan order on a tie), and that sample's plane -
    /// its range and normal - is re-intersected with the output pixel's own
    /// ray, whose centre is the corner the four samples share: exact on a
    /// planar surface, for any lens. Where the ray meets that plane too
    /// obliquely to trust, the sample's range is kept as it is.
    pub fn halved(&self, k: &Intrinsics) -> DepthMap {
        assert_eq!((k.width, k.height), (self.width, self.height), "halved: the calibration must describe this map's size");
        let (w, h) = (self.width / 2, self.height / 2);
        let mut out = DepthMap::empty(w, h);
        let sw = self.width as usize;
        for y in 0..h as usize {
            for x in 0..w as usize {
                let mut pick: Option<usize> = None;
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let c = (2 * y + dy) * sw + 2 * x + dx;
                    if self.range[c] > 0.0 && pick.is_none_or(|p| self.conf[c] > self.conf[p]) {
                        pick = Some(c);
                    }
                }
                let Some(c) = pick else { continue };
                let n = [self.normal[3 * c] as f64, self.normal[3 * c + 1] as f64, self.normal[3 * c + 2] as f64];
                let mut r = self.range[c];
                let child = [(c % sw) as f64 + 0.5, (c / sw) as f64 + 0.5];
                let centre = [2.0 * x as f64 + 1.0, 2.0 * y as f64 + 1.0];
                if let (Some(dc), Some(dp)) = (k.unproject(child), k.unproject(centre)) {
                    let den = sfm::linalg::dot(n, dp);
                    if den < -1e-3 {
                        let rp = sfm::linalg::dot(n, dc) * r as f64 / den;
                        if rp > 0.0 {
                            r = rp as f32;
                        }
                    }
                }
                let o = y * w as usize + x;
                out.range[o] = r;
                out.normal[3 * o..3 * o + 3].copy_from_slice(&self.normal[3 * c..3 * c + 3]);
                out.conf[o] = self.conf[c];
            }
        }
        out
    }
}

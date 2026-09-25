// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Photographs to a splat training set, with no learned model anywhere in
//! the loop: structure from motion ([`sfm`]) recovers every camera's pose,
//! each sensor's calibration through its REAL lens and a sparse coloured
//! point cloud, and every photograph becomes a target exactly as it was
//! recorded - its own pixels, imaged through that lens.
//!
//! Nothing is resampled into an idealized camera. The renderer the fit uses
//! evaluates every gaussian along each pixel's own ray through the lens
//! model, so undistorting a photograph could only lose what it recorded:
//! interpolation spreads each pixel over its neighbours and blurs exactly the
//! detail a fit is trying to recover, and the corners a pinhole frame cannot
//! reach are thrown away. A lower training resolution is reached by exact
//! halvings (a 2x2 box, which is the area average at that scale and moves
//! every pixel coordinate by exactly one half), never by a resample at an
//! arbitrary factor.
//!
//! Swedish Embedded AB implements photogrammetry pipelines - from a folder of
//! photographs to calibrated cameras and a radiance field - for its clients.
//! If your team needs that, you can procure our services by sending an email
//! to info@swedishembedded.com.

use imaging::Rgb8;
use sfm::incremental::{reconstruct, Photo, Reconstruction, SfmCfg, SfmError};
use splat::opt::TargetView;
use splat::types::{Camera, Splats};

/// What a fit needs, recovered from photographs alone.
pub struct TrainingSet {
    /// One per REGISTERED photograph, in input order.
    pub targets: Vec<TargetView>,
    /// Input index of each target.
    pub source: Vec<usize>,
    /// The starting scene, from the structure-from-motion points.
    pub init: Splats,
    /// The full reconstruction, for reporting.
    pub sfm: Reconstruction,
}

impl TrainingSet {
    /// The same set landed upright: structure from motion leaves the scene in
    /// its first camera's frame, however that camera was held, and a viewer
    /// opening the result expects the ground to be down.
    pub fn upright(self) -> TrainingSet {
        let cams: Vec<Camera> = self.targets.iter().map(|t| t.cam).collect();
        let (init, cams) = splat::orient::upright(&self.init, &cams);
        let targets = self
            .targets
            .into_iter()
            .zip(cams)
            .map(|(t, cam)| TargetView { cam, ..t })
            .collect();
        TrainingSet { targets, init, ..self }
    }
}

/// A photograph as a target: its own pixels in [0,1], through its camera,
/// with the pixels its lens has no ray for (outside a fisheye's image
/// circle) masked out rather than supervised against nothing.
pub fn target(photo: &Rgb8, cam: Camera, sensor: usize) -> TargetView {
    let rgb: Vec<f32> = photo.px.iter().map(|&v| v as f32 / 255.0).collect();
    let k = cam.intrinsics();
    let (w, h) = (photo.w as usize, photo.h as usize);
    let mask: Vec<f32> = backend_cpu::par::map_f32(w * h, |i| {
        let px = [(i % w) as f64 + 0.5, (i / w) as f64 + 0.5];
        if k.unproject(px).is_some() { 1.0 } else { 0.0 }
    });
    let t = TargetView::new(cam, rgb).with_sensor(sensor);
    if mask.iter().all(|&m| m == 1.0) { t } else { t.with_mask(mask) }
}

/// Recover cameras from `photos` and build the training set `halvings`
/// exact halvings below the photographs' own resolution.
/// `opacity` is what every starting gaussian gets.
pub fn training_set(photos: &[Rgb8], halvings: u32, opacity: f32, cfg: &SfmCfg) -> Result<TrainingSet, SfmError> {
    let views: Vec<Photo> = photos.iter().map(|p| Photo { width: p.w, height: p.h, rgb: &p.px, sensor: 0, focal_px: None }).collect();
    let rec = reconstruct(&views, cfg)?;
    let mut targets = Vec::new();
    let mut source = Vec::new();
    for (i, pose) in rec.poses.iter().enumerate() {
        let Some(pose) = pose else { continue };
        let s = rec.sensor[i];
        let m = pose.c2w();
        let cam = Camera::with_intrinsics(std::array::from_fn(|k| m[k] as f32), &rec.intrinsics[s]);
        let mut t = target(&photos[i], cam, s);
        for _ in 0..halvings {
            t = t.half();
        }
        targets.push(t);
        source.push(i);
    }
    let xyz: Vec<f32> = rec.points.iter().flat_map(|p| p.xyz.map(|v| v as f32)).collect();
    let rgb: Vec<f32> = rec.points.iter().flat_map(|p| p.rgb).collect();
    let init = splat::init::from_points(&xyz, &rgb, opacity);
    Ok(TrainingSet { targets, source, init, sfm: rec })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A photograph is supervised as recorded: its pixels unchanged, every
    /// one of them whose ray the lens can trace - and outside a fisheye's
    /// image circle, none.
    #[test]
    fn a_target_is_the_photograph_itself_masked_where_the_lens_sees_nothing() {
        let (w, h) = (40u32, 30u32);
        let px: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 % 251) as u8).collect();
        let img = Rgb8 { w, h, px: px.clone() };
        let k = camera::Intrinsics { lens: camera::Lens::radial(-0.05, 0.01), ..camera::Intrinsics::pinhole(30.0, w, h) };
        let t = target(&img, Camera::with_intrinsics(std::array::from_fn(|i| if i % 5 == 0 { 1.0 } else { 0.0 }), &k), 0);
        for i in 0..px.len() {
            assert!((t.rgb[i] - px[i] as f32 / 255.0).abs() < 1e-6, "pixel value {i}");
        }
        assert!(t.mask.is_none(), "every pixel of a mild barrel lens has a ray");
        // a fisheye whose image circle is smaller than the frame: its lens
        // stops at 60 degrees off axis
        let fish = camera::Intrinsics { lens: camera::Lens::Fisheye { k: [0.0; 4] }, ..camera::Intrinsics::pinhole(12.0, w, h) };
        let t = target(&img, Camera::with_intrinsics(std::array::from_fn(|i| if i % 5 == 0 { 1.0 } else { 0.0 }), &fish), 0);
        let mask = t.mask.expect("pixels beyond the fisheye's reach are masked");
        assert_eq!(mask[(h / 2 * w + w / 2) as usize], 1.0);
        assert!(mask.contains(&0.0), "a corner past 180 degrees has no ray");
    }
}

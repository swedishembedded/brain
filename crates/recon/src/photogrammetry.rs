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
//! [`dense`] adds what the sparse points cannot give a fit: multi-view stereo
//! ([`mvs`]) measures every photograph's range, surface normal and their
//! confidence per pixel, through the same lens, which become each target's
//! geometry priors, and fuses them into a dense starting scene of thin
//! gaussians lying in the measured surfaces.
//!
//! Swedish Embedded AB implements photogrammetry pipelines - from a folder of
//! photographs to calibrated cameras and a radiance field - for its clients.
//! If your team needs that, you can procure our services by sending an email
//! to info@swedishembedded.com.

use gpu_core::Gpu;
use imaging::Rgb8;
use sfm::incremental::{reconstruct, Photo, Reconstruction, SfmCfg, SfmError};
use splat::opt::TargetView;
use splat::types::{Camera, Splats};

/// A rigid change of frame `p' = r (p - centre)`, `r` row-major.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    pub r: [f64; 9],
    pub centre: [f64; 3],
}

impl Frame {
    pub const IDENTITY: Frame = Frame { r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], centre: [0.0; 3] };

    pub fn apply(&self, p: [f64; 3]) -> [f64; 3] {
        let d = [p[0] - self.centre[0], p[1] - self.centre[1], p[2] - self.centre[2]];
        std::array::from_fn(|i| self.r[i * 3] * d[0] + self.r[i * 3 + 1] * d[1] + self.r[i * 3 + 2] * d[2])
    }

    /// `next` after `self`: `next.apply(self.apply(p))`.
    pub fn then(&self, next: &Frame) -> Frame {
        let r = std::array::from_fn(|k| {
            let (i, j) = (k / 3, k % 3);
            (0..3).map(|m| next.r[i * 3 + m] * self.r[m * 3 + j]).sum()
        });
        // r1 (p - c1) - c2 = r1 (p - c1 - r1^T c2)
        let back: [f64; 3] = std::array::from_fn(|j| (0..3).map(|i| self.r[i * 3 + j] * next.centre[i]).sum());
        Frame { r, centre: std::array::from_fn(|k| self.centre[k] + back[k]) }
    }
}

/// What a fit needs, recovered from photographs alone.
pub struct TrainingSet {
    /// One per REGISTERED photograph, in input order.
    pub targets: Vec<TargetView>,
    /// Input index of each target.
    pub source: Vec<usize>,
    /// The starting scene, from the structure-from-motion points.
    pub init: Splats,
    /// The full reconstruction, in its own frame.
    pub sfm: Reconstruction,
    /// From the reconstruction's frame to the targets' and `init`'s.
    pub frame: Frame,
}

impl TrainingSet {
    /// The same set landed upright: structure from motion leaves the scene in
    /// its first camera's frame, however that camera was held, and a viewer
    /// opening the result expects the ground to be down.
    pub fn upright(self) -> TrainingSet {
        let cams: Vec<Camera> = self.targets.iter().map(|t| t.cam).collect();
        let mats: Vec<[f64; 16]> = cams.iter().map(|c| std::array::from_fn(|i| c.c2w[i] as f64)).collect();
        let (r, centre) = splat::orient::frame_from_cameras(&mats, -1.0);
        let init = splat::orient::apply(&self.init, &r, &centre);
        let targets = self
            .targets
            .into_iter()
            .zip(&mats)
            .map(|(t, m)| {
                let c2w = splat::orient::transform_c2w(m, &r, &centre);
                TargetView { cam: Camera { c2w: std::array::from_fn(|i| c2w[i] as f32), ..t.cam }, ..t }
            })
            .collect();
        TrainingSet { targets, init, frame: self.frame.then(&Frame { r, centre }), ..self }
    }

    /// The same set with multi-view stereo's geometry: every target carries
    /// its measured range, normal and confidence as priors, and `init` is
    /// the dense scene fused from them. `photos` are the photographs the set
    /// was built from, in input order; `gpu` must hold [`mvs::PIPELINES`] at
    /// `ks`.
    pub fn densify(mut self, gpu: &Gpu, ks: &mvs::Kernels, photos: &[Rgb8], cfg: &DenseCfg) -> Result<(TrainingSet, DenseReport), mvs::MvsError> {
        let full: Vec<Camera> = self
            .targets
            .iter()
            .zip(&self.source)
            .map(|(t, &i)| Camera { c2w: t.cam.c2w, shutter: t.cam.shutter, ..Camera::with_intrinsics(t.cam.c2w, &self.sfm.intrinsics[self.sfm.sensor[i]]) })
            .collect();
        let mut view_of = vec![None; self.sfm.poses.len()];
        for (v, &i) in self.source.iter().enumerate() {
            view_of[i] = Some(v);
        }
        let tracks: Vec<mvs::Track> = mvs::select::tracks_from_sfm(&self.sfm, &view_of)
            .into_iter()
            .map(|t| mvs::Track { xyz: self.frame.apply(t.xyz), ..t })
            .collect();
        let rgb: Vec<&Rgb8> = self.source.iter().map(|&i| &photos[i]).collect();
        let (init, report) = dense(gpu, ks, &full, &rgb, &tracks, &mut self.targets, cfg)?;
        Ok((TrainingSet { init, ..self }, report))
    }
}

/// Multi-view stereo, fusion and the dense starting scene, as [`dense`]
/// runs them.
#[derive(Clone, Debug, Default)]
pub struct DenseCfg {
    pub stereo: mvs::StereoCfg,
    pub fuse: mvs::FuseCfg,
    pub init: mvs::SplatInit,
}

/// What [`dense`] measured.
#[derive(Clone, Debug)]
pub struct DenseReport {
    /// Fraction of each target's pixels with a range prior.
    pub coverage: Vec<f64>,
    /// Points in the fused cloud, one gaussian each.
    pub points: usize,
    pub timings: mvs::Timings,
}

/// Measure every view by multi-view stereo, attach the measurements to
/// `targets` as priors at each target's own resolution, and fuse them into a
/// dense scene.
///
/// `cams` are the views' cameras at the photographs' full resolution, in the
/// targets' frame; `tracks` are the structure-from-motion points in that
/// frame, which choose each view's sources and bound its search. A target
/// imaged at a lower resolution than the stereo's finest level gets the map
/// halved exactly (`mvs::DepthMap::halved`, which keeps edges); one imaged
/// finer than it is an error, since a map cannot be upsampled without
/// inventing geometry.
pub fn dense(
    gpu: &Gpu,
    ks: &mvs::Kernels,
    cams: &[Camera],
    photos: &[&Rgb8],
    tracks: &[mvs::Track],
    targets: &mut [TargetView],
    cfg: &DenseCfg,
) -> Result<(Splats, DenseReport), mvs::MvsError> {
    assert_eq!(cams.len(), targets.len(), "dense: one camera per target");
    assert_eq!(photos.len(), targets.len(), "dense: one photograph per target");
    let views: Vec<mvs::View> = cams.iter().zip(photos).map(|(c, p)| mvs::View { cam: *c, rgb: &p.px }).collect();
    let st = mvs::depth_maps(gpu, ks, &views, tracks, &cfg.stereo)?;
    let fused = mvs::fuse(gpu, ks, &st.cams, &st.depth, &st.rgb, &cfg.fuse)?;
    let mut coverage = Vec::with_capacity(targets.len());
    for ((t, map), cam) in targets.iter_mut().zip(&st.depth).zip(&st.cams) {
        let (mut map, mut cam) = (map.clone(), *cam);
        while map.width > t.cam.width {
            map = map.halved(&cam.intrinsics());
            cam = Camera { fx: cam.fx * 0.5, fy: cam.fy * 0.5, cx: cam.cx * 0.5, cy: cam.cy * 0.5, width: cam.width / 2, height: cam.height / 2, ..cam };
        }
        if (map.width, map.height) != (t.cam.width, t.cam.height) {
            return Err(mvs::MvsError::Mismatch(format!(
                "a {}x{} target cannot take a {}x{} depth map: stereo must run at the target's resolution or finer",
                t.cam.width, t.cam.height, map.width, map.height
            )));
        }
        coverage.push(map.coverage());
        t.depth = Some(map.range);
        t.depth_conf = Some(map.conf);
        t.normals = Some(map.normal);
    }
    let init = mvs::to_splats(&fused, &cfg.init);
    Ok((init, DenseReport { coverage, points: fused.len(), timings: st.timings }))
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
    Ok(TrainingSet { targets, source, init, sfm: rec, frame: Frame::IDENTITY })
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
        // an equidistant fisheye whose image circle is smaller than the
        // frame: 180 degrees off axis is pi f = 18.8 px from the centre, and
        // the corners are 25 px out
        let fish = camera::Intrinsics { lens: camera::Lens::Fisheye { k: [0.0; 4] }, ..camera::Intrinsics::pinhole(6.0, w, h) };
        let t = target(&img, Camera::with_intrinsics(std::array::from_fn(|i| if i % 5 == 0 { 1.0 } else { 0.0 }), &fish), 0);
        let mask = t.mask.expect("pixels beyond the fisheye's reach are masked");
        assert_eq!(mask[(h / 2 * w + w / 2) as usize], 1.0);
        assert!(mask.contains(&0.0), "a corner past 180 degrees has no ray");
    }
}

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Photographs to a splat training set, with no learned model anywhere in
//! the loop: structure from motion ([`sfm`]) recovers the cameras, one
//! shared calibration and a sparse coloured point cloud; each photograph is
//! then resampled to an ideal pinhole camera at the training resolution, and
//! the point cloud becomes the starting scene.
//!
//! The rasterizer is a pinhole renderer, so lens distortion is removed from
//! the TARGETS rather than modelled in the render. Pixels of the pinhole
//! frame that fall outside the original photograph (the corners, under
//! pincushion distortion) carry no measurement and are masked out of the
//! loss.
//!
//! Swedish Embedded AB implements photogrammetry pipelines - from a folder of
//! photographs to calibrated cameras and a radiance field - for its clients.
//! If your team needs that, you can procure our services by sending an email
//! to info@swedishembedded.com.

use camera::Intrinsics;
use imaging::Rgb8;
use sfm::camera::Pose;
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

/// Resample `img` (taken through `k`) to a pinhole camera of `width x height`
/// with the same field of view along the long side. Returns interleaved RGB
/// in [0,1], the validity mask, and the pinhole intrinsics `(f, cx, cy)`.
pub fn undistort(img: &Rgb8, k: &Intrinsics, width: u32, height: u32) -> (Vec<f32>, Vec<f32>, (f32, f32, f32)) {
    let s = width as f64 / k.width as f64;
    let (f, cx, cy) = (k.fx * s, k.cx * s, k.cy * s);
    let (w, h) = (width as usize, height as usize);
    let mut rgb = vec![0.0f32; w * h * 3];
    let mut mask = vec![0.0f32; w * h];
    let (iw, ih) = (img.w as usize, img.h as usize);
    for y in 0..h {
        for x in 0..w {
            let Some(p) = k.project([(x as f64 + 0.5 - cx) / f, (y as f64 + 0.5 - cy) / f, 1.0]) else { continue };
            // bilinear on pixel centres
            let (fx, fy) = (p[0] - 0.5, p[1] - 0.5);
            if fx < 0.0 || fy < 0.0 || fx > (iw - 1) as f64 || fy > (ih - 1) as f64 {
                continue;
            }
            let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
            let (x1, y1) = ((x0 + 1).min(iw - 1), (y0 + 1).min(ih - 1));
            let (tx, ty) = ((fx - x0 as f64) as f32, (fy - y0 as f64) as f32);
            let o = (y * w + x) * 3;
            for c in 0..3 {
                let at = |xx: usize, yy: usize| img.px[(yy * iw + xx) * 3 + c] as f32 / 255.0;
                let top = at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx;
                let bot = at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx;
                rgb[o + c] = top * (1.0 - ty) + bot * ty;
            }
            mask[y * w + x] = 1.0;
        }
    }
    (rgb, mask, (f as f32, cx as f32, cy as f32))
}

/// The splat camera of an SfM pose with pinhole intrinsics `(f, cx, cy)`.
pub fn camera(pose: &Pose, fcc: (f32, f32, f32), width: u32, height: u32) -> Camera {
    let m = pose.c2w();
    Camera { c2w: std::array::from_fn(|i| m[i] as f32), fx: fcc.0, fy: fcc.0, cx: fcc.1, cy: fcc.2, width, height }
}

/// Recover cameras from `photos` and build the training set at `width`
/// pixels across (the height follows the photographs' aspect).
/// `opacity` is what every starting gaussian gets (3DGS starts at 0.1).
pub fn training_set(photos: &[Rgb8], width: u32, opacity: f32, cfg: &SfmCfg) -> Result<TrainingSet, SfmError> {
    let views: Vec<Photo> = photos.iter().map(|p| Photo { width: p.w, height: p.h, rgb: &p.px, sensor: 0, focal_px: None }).collect();
    let rec = reconstruct(&views, cfg)?;
    let k = rec.intrinsics[0];
    let height = ((width as f64 * k.height as f64 / k.width as f64).round() as u32).max(1);
    // Downscale first so the pinhole resample reads a band-limited source.
    let factor = (k.width as f64 / width as f64).floor().max(1.0) as u32;
    let mut targets = Vec::new();
    let mut source = Vec::new();
    for (i, pose) in rec.poses.iter().enumerate() {
        let Some(pose) = pose else { continue };
        let (small, ks) = if factor > 1 { box_down(&photos[i], factor, &k) } else { (photos[i].clone(), k) };
        let (rgb, mask, fcc) = undistort(&small, &ks, width, height);
        targets.push(TargetView::new(camera(pose, fcc, width, height), rgb).with_mask(mask));
        source.push(i);
    }
    let xyz: Vec<f32> = rec.points.iter().flat_map(|p| p.xyz.map(|v| v as f32)).collect();
    let rgb: Vec<f32> = rec.points.iter().flat_map(|p| p.rgb).collect();
    let mut init = splat::init::from_points(&xyz, &rgb, opacity);
    let cams: Vec<Camera> = targets.iter().map(|t| t.cam).collect();
    splat::init::floor_to_pixels(&mut init, &cams, 1.0);
    Ok(TrainingSet { targets, source, init, sfm: rec })
}

/// Average `factor x factor` blocks, and the intrinsics of the result.
fn box_down(img: &Rgb8, factor: u32, k: &Intrinsics) -> (Rgb8, Intrinsics) {
    let (w, h) = (img.w / factor, img.h / factor);
    let mut px = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let mut s = 0u32;
                for dy in 0..factor {
                    for dx in 0..factor {
                        s += img.px[(((y * factor + dy) * img.w + x * factor + dx) * 3 + c) as usize] as u32;
                    }
                }
                px[((y * w + x) * 3 + c) as usize] = (s / (factor * factor)) as u8;
            }
        }
    }
    let s = 1.0 / factor as f64;
    let ks = Intrinsics { fx: k.fx * s, fy: k.fy * s, cx: k.cx * s, cy: k.cy * s, width: w, height: h, ..*k };
    (Rgb8 { w, h, px }, ks)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With no distortion and no change of size the pinhole resample is the
    /// photograph itself, every pixel valid; with pincushion distortion the
    /// corners of the pinhole frame fall outside the photograph and are
    /// masked rather than invented.
    #[test]
    fn undistortion_is_exact_at_identity_and_masks_what_it_cannot_see() {
        let (w, h) = (40u32, 30u32);
        let px: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 % 251) as u8).collect();
        let img = Rgb8 { w, h, px: px.clone() };
        let k = Intrinsics::pinhole(0.8 * w.max(h) as f64, w, h);
        let (rgb, mask, _) = undistort(&img, &k, w, h);
        for i in 0..px.len() {
            assert!((rgb[i] - px[i] as f32 / 255.0).abs() < 1e-5, "pixel value {i}");
        }
        assert!(mask.iter().all(|&m| m == 1.0));
        let pincushion = Intrinsics { lens: camera::Lens::radial(0.2, 0.0), ..k };
        let (_, mask, _) = undistort(&img, &pincushion, w, h);
        assert_eq!(mask[0], 0.0, "a corner outside the photograph must be masked");
        assert_eq!(mask[(h / 2 * w + w / 2) as usize], 1.0);
    }
}

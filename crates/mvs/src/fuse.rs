// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Depth-map fusion: every view's filtered measurements, merged into one
//! world-space point per surface sample.
//!
//! A measurement joins another when the two agree - carried into each
//! other's view they land within [`FuseCfg::max_reproj_px`] pixels and
//! [`FuseCfg::max_rel_range`] in range, with normals within
//! [`FuseCfg::max_normal_deg`] (Schönberger et al., ECCV 2016, §5). The point
//! is emitted ONCE, by the view that samples it most finely, averaging every
//! agreeing measurement: range along that view's ray (so the point never
//! moves sideways off it), normal, colour and confidence, with the number of
//! agreeing views as its support and the owner's pixel footprint at that
//! range as its radius. Unlike a greedy sequential merge this is a pure
//! gather per view pair (`mvs_fuse_acc.wgsl`, `mvs_fuse_out.wgsl`), so it runs
//! on the device with no atomics and gives the same cloud in any order.

use gpu_core::Gpu;
use splat::types::Camera;

use crate::depth::DepthMap;
use crate::stereo::{c2w_words, cam_words, rays_step, relative};
use crate::{pose_of, Kernels, MvsError};

/// When measurements of different views are one surface point.
#[derive(Clone, Debug, PartialEq)]
pub struct FuseCfg {
    /// Largest reprojection error between agreeing measurements, pixels.
    pub max_reproj_px: f32,
    /// Largest relative range disagreement.
    pub max_rel_range: f32,
    /// Largest angle between agreeing normals, degrees.
    pub max_normal_deg: f32,
    /// Fewest views, the emitting one included, a point needs.
    pub min_support: u32,
}

impl Default for FuseCfg {
    fn default() -> Self {
        FuseCfg { max_reproj_px: 2.0, max_rel_range: 0.01, max_normal_deg: 20.0, min_support: 3 }
    }
}

/// A fused point cloud, SoA.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Fused {
    /// `[N*3]` world position.
    pub xyz: Vec<f32>,
    /// `[N*3]` unit world normal, facing the views that saw it.
    pub normal: Vec<f32>,
    /// `[N*3]` colour in [0, 1].
    pub rgb: Vec<f32>,
    /// `[N]` mean confidence of the fused measurements, in [0, 1].
    pub conf: Vec<f32>,
    /// `[N]` world size of the emitting pixel's footprint at the point.
    pub radius: Vec<f32>,
    /// `[N]` views that agree on the point, the emitting one included.
    pub support: Vec<u32>,
}

impl Fused {
    pub fn len(&self) -> usize {
        self.conf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.conf.is_empty()
    }
}

/// Planar device layout of a map: range, normal (3), confidence.
fn map_planes(d: &DepthMap, plane: usize) -> Vec<f32> {
    let n = d.range.len();
    let mut v = vec![0.0f32; 5 * plane];
    v[..n].copy_from_slice(&d.range);
    for i in 0..n {
        for c in 0..3 {
            v[(1 + c) * plane + i] = d.normal[3 * i + c];
        }
    }
    v[4 * plane..4 * plane + n].copy_from_slice(&d.conf);
    v
}

/// Fuse the maps of `cams` (at the maps' resolution; `rgb` interleaved in
/// [0, 1] at the same size, as [`crate::Stereo`] returns them).
pub fn fuse(gpu: &Gpu, ks: &Kernels, cams: &[Camera], depth: &[DepthMap], rgb: &[Vec<f32>], cfg: &FuseCfg) -> Result<Fused, MvsError> {
    if cams.len() != depth.len() || cams.len() != rgb.len() {
        return Err(MvsError::Mismatch(format!("fuse: {} cameras, {} maps, {} images", cams.len(), depth.len(), rgb.len())));
    }
    for (v, (c, d)) in cams.iter().zip(depth).enumerate() {
        let n = c.width as usize * c.height as usize;
        if (d.width, d.height) != (c.width, c.height) || d.range.len() != n || d.normal.len() != 3 * n || d.conf.len() != n || rgb[v].len() != 3 * n {
            return Err(MvsError::Mismatch(format!("fuse: view {v}'s map or image does not match its {}x{} camera", c.width, c.height)));
        }
    }
    let poses: Vec<_> = cams.iter().map(pose_of).collect();
    let ks_cam: Vec<_> = cams.iter().map(Camera::intrinsics).collect();
    let planes: Vec<usize> = cams.iter().map(|c| (c.width as usize * c.height as usize).div_ceil(64) * 64).collect();
    let maps: Vec<_> = depth.iter().zip(&planes).map(|(d, &p)| gpu.storage_init("mvs.fuse.map", &map_planes(d, p))).collect();
    let colours: Vec<_> = rgb
        .iter()
        .zip(&planes)
        .map(|(img, &p)| {
            let mut v = vec![0.0f32; 3 * p];
            for (i, px) in img.chunks_exact(3).enumerate() {
                for c in 0..3 {
                    v[c * p + i] = px[c];
                }
            }
            gpu.storage_init("mvs.fuse.rgb", &v)
        })
        .collect();
    let max_plane = planes.iter().copied().max().unwrap_or(64) as u64;
    let rays = gpu.storage(3 * max_plane);
    let acc = gpu.storage(10 * max_plane);
    let out = gpu.storage(12 * max_plane);
    let cos_normal = cfg.max_normal_deg.to_radians().cos();
    let mut fused = Fused::default();
    for i in 0..cams.len() {
        if depth[i].coverage() == 0.0 {
            continue;
        }
        let (w, h, pl) = (cams[i].width, cams[i].height, planes[i] as u32);
        let mut steps = vec![rays_step(gpu, ks, &ks_cam[i], pl, &rays)];
        for j in (0..cams.len()).filter(|&j| j != i && depth[j].coverage() > 0.0) {
            let (r, t) = relative(&poses[i], &poses[j]);
            let mut p = vec![w, h, pl, planes[j] as u32, (j < i) as u32, 0, 0, 0];
            p.extend_from_slice(&[gpu_core::f(cfg.max_reproj_px), gpu_core::f(cfg.max_rel_range), gpu_core::f(cos_normal), 0]);
            p.extend_from_slice(&ks_cam[i].device_lens());
            p.extend_from_slice(&cam_words(&r, t, &ks_cam[j]));
            steps.push(gpu.step(ks.fuse_acc, &[&maps[i], &rays, &maps[j], &colours[j], &acc], &p, w * h));
        }
        let mut p = vec![w, h, pl, cfg.min_support];
        p.extend_from_slice(&c2w_words(&poses[i], &ks_cam[i]));
        steps.push(gpu.step(ks.fuse_out, &[&maps[i], &rays, &colours[i], &acc, &out], &p, w * h));
        gpu.submit(&[&acc], &steps);
        let o = gpu.read(&out, 12 * pl as usize);
        let pl = pl as usize;
        for px in 0..(w * h) as usize {
            let support = o[11 * pl + px];
            if support <= 0.0 {
                continue;
            }
            fused.xyz.extend((0..3).map(|c| o[c * pl + px]));
            fused.normal.extend((3..6).map(|c| o[c * pl + px]));
            fused.rgb.extend((6..9).map(|c| o[c * pl + px]));
            fused.conf.push(o[9 * pl + px]);
            fused.radius.push(o[10 * pl + px]);
            fused.support.push(support as u32);
        }
    }
    Ok(fused)
}

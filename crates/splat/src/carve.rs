// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Free-space carving: which gaussians sit where the cameras saw nothing.
//!
//! A camera that measured a surface at range `r` along a ray observed the
//! segment before `r` to be empty. The photometric objective does not
//! know that: a translucent gaussian floating in front of a surface can
//! serve one view's colour while every other view looks straight through
//! the space it occupies. [`carve`] counts, per gaussian, the views that saw
//! empty space where it sits (it lies wholly in front of every surface they
//! measured around its projection) and the views that saw it on their
//! surface (`splat_carve.wgsl`). The measurements are whatever the caller
//! trusts per view: a range prior from stereo, or the scene's own rendered
//! surface in the other views.
//!
//! Swedish Embedded AB implements reconstruction whose geometry is held to
//! what every camera observed. If your team needs expertise in multi-view
//! geometry then you can procure our services by sending an email to
//! info@swedishembedded.com.

use gpu_core::{DeviceBuffer, Gpu};

use crate::renderer::{ray_view_params, GpuSplats, RAY_VIEW_WORDS};
use crate::types::{Camera, RenderOpts};
use crate::Kernels;

/// What a set of views measured, on the device: each view's record (as
/// the ray renderer takes it) and its measured range per pixel (0 = none).
pub struct Observations {
    views: DeviceBuffer,
    ranges: DeviceBuffer,
    plane: usize,
    n: usize,
}

impl Observations {
    /// `ranges[k]` is view `k`'s measured range per pixel, `cams[k]`'s size;
    /// the views need not share a size.
    pub fn new(gpu: &Gpu, cams: &[Camera], ranges: &[Vec<f32>]) -> Observations {
        assert_eq!(cams.len(), ranges.len(), "one range map per view");
        let plane = cams.iter().map(|c| (c.width * c.height) as usize).max().unwrap_or(0);
        let mut words = Vec::with_capacity(RAY_VIEW_WORDS * cams.len());
        let mut flat = vec![0.0f32; plane * cams.len()];
        for (k, (c, r)) in cams.iter().zip(ranges).enumerate() {
            assert_eq!(r.len(), (c.width * c.height) as usize, "view {k}: its range map is not its size");
            words.extend_from_slice(&ray_view_params(0, c, &RenderOpts { ray: true, ..Default::default() }));
            flat[k * plane..k * plane + r.len()].copy_from_slice(r);
        }
        let views = gpu.storage(words.len().max(RAY_VIEW_WORDS) as u64);
        if !words.is_empty() {
            gpu.write(&views, &words);
        }
        Observations { views, ranges: gpu.storage_init("carve.ranges", if flat.is_empty() { &[0.0] } else { &flat }), plane, n: cams.len() }
    }
}

/// Per gaussian of `s`, `[violations, supports]`: views that saw empty space
/// where it sits, views that saw it on their surface, with `margin` the
/// relative range tolerance.
pub fn carve(gpu: &Gpu, ks: Kernels, s: &GpuSplats, obs: &Observations, margin: f32) -> Vec<[f32; 2]> {
    let out = gpu.storage((2 * s.n.max(1)) as u64);
    let step = gpu.step(
        ks.splat_carve,
        &[&s.means, &s.scales, &obs.views, &obs.ranges, &out],
        &[s.n as u32, obs.n as u32, obs.plane as u32, 0, gpu_core::f(margin), 0, 0, 0],
        s.n as u32,
    );
    gpu.submit(&[], &[step]);
    let v = gpu.read(&out, 2 * s.n);
    gpu_core::reclaiming(gpu, || drop(out));
    v.chunks_exact(2).map(|c| [c[0], c[1]]).collect()
}

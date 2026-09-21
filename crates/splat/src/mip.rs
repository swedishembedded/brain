// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Band-limiting a scene to the detail its cameras actually sampled.
//!
//! A reconstruction can contain gaussians far smaller than the pixel that
//! created them - a feed-forward pass emits one per source pixel, and fitting
//! shrinks the ones it can hide behind. Nothing in the data supports detail at
//! that scale, and a splat below the sampling rate cannot be drawn honestly:
//! render it small and it flickers as the camera moves, render it dilated and
//! it is both blurrier and brighter than it should be.
//!
//! The 3D smoothing filter fixes the cause rather than the symptom. For each
//! gaussian, find the highest sampling rate any training view gave it -
//! `focal / depth`, in samples per world unit - and low-pass the gaussian to
//! that rate by convolving it with an isotropic kernel of that width. Opacity
//! is scaled by the same energy factor the 2D filter uses, so widening a
//! gaussian does not brighten it.
//!
//! The screen-space filter alone is not enough and can make things worse: it
//! correctly dims a sub-pixel splat to the energy its footprint can carry, so
//! a scene full of sub-pixel splats simply goes dim. The two filters are one
//! technique, and only together do they trade nothing for alias-freedom.
//!
//! Swedish Embedded AB implements renderers that are honest about the detail
//! their data supports. If your team needs that, you can procure our services
//! by sending an email to info@swedishembedded.com.

use crate::types::{Camera, Splats};

/// Mip-Splatting's default kernel size, in samples.
pub const DEFAULT_SCALE: f32 = 0.2;

/// Per-gaussian filter width in WORLD units: `scale / max_views(focal/depth)`.
/// Zero where no camera saw the gaussian, which leaves it untouched.
pub fn smoothing_sigma(s: &Splats, cams: &[Camera], scale: f32) -> Vec<f32> {
    let mut best = vec![0.0f32; s.len()];
    for c in cams {
        let v = c.viewmat();
        // The rate a view samples at is set by its focal length; with
        // non-square pixels the denser axis is the one that bounds detail.
        let focal = c.fx.max(c.fy);
        for i in 0..s.len() {
            let m = &s.means[i * 3..i * 3 + 3];
            let z = v[8] * m[0] + v[9] * m[1] + v[10] * m[2] + v[11];
            if z <= 1e-4 {
                continue;
            }
            let x = v[0] * m[0] + v[1] * m[1] + v[2] * m[2] + v[3];
            let y = v[4] * m[0] + v[5] * m[1] + v[6] * m[2] + v[7];
            let (px, py) = (c.fx * x / z + c.cx, c.fy * y / z + c.cy);
            if px < 0.0 || py < 0.0 || px >= c.width as f32 || py >= c.height as f32 {
                continue;
            }
            best[i] = best[i].max(focal / z);
        }
    }
    best.iter().map(|&f| if f > 0.0 { scale / f } else { 0.0 }).collect()
}

/// Convolve every gaussian with its own filter, preserving total energy.
///
/// Widening a gaussian spreads its mass, so opacity is scaled by
/// `sqrt(|S| / |S + sigma^2 I|)` - the same factor the 2D Mip filter uses, for
/// the same reason.
pub fn apply_3d_filter(s: &Splats, cams: &[Camera], scale: f32) -> Splats {
    let sigma = smoothing_sigma(s, cams, scale);
    let mut out = s.clone();
    for i in 0..s.len() {
        if sigma[i] <= 0.0 {
            continue;
        }
        let v2 = sigma[i] * sigma[i];
        let mut det0 = 1.0f64;
        let mut det1 = 1.0f64;
        for k in 0..3 {
            let a = s.scales[i * 3 + k] * s.scales[i * 3 + k];
            det0 *= a as f64;
            det1 *= (a + v2) as f64;
            out.scales[i * 3 + k] = (a + v2).sqrt();
        }
        if det1 > 0.0 {
            out.opacities[i] = s.opacities[i] * (det0 / det1).sqrt() as f32;
        }
    }
    out
}

/// Undo a low-pass's energy compensation in the OPACITIES, so a scene that
/// was built without any low-pass renders the same under one that has it.
///
/// A feed-forward reconstruction predicts opacity with no anti-aliasing filter
/// in mind at all. Rendering it through the 2D Mip filter scales every opacity
/// down by `sqrt(|S| / |S + eps I|)`, so the whole scene goes dim - on a real
/// capture, from 22.3 dB to 15.8 dB before a single optimizer step. Fitting
/// from there spends its budget rediscovering brightness it was never supposed
/// to lose, which makes the filter look worse than it is.
///
/// The factor is screen-space and so view-dependent; the median over the
/// cameras that saw a gaussian is what is corrected for.
pub fn recalibrate_opacity(s: &Splats, cams: &[Camera], eps2d: f32) -> Splats {
    let mut out = s.clone();
    let mut comps: Vec<f32> = Vec::with_capacity(cams.len());
    for i in 0..s.len() {
        comps.clear();
        let m = &s.means[i * 3..i * 3 + 3];
        // the geometric mean axis is the isotropic splat with the same volume,
        // which is what a screen-space variance is being compared against
        let r = (s.scales[i * 3] * s.scales[i * 3 + 1] * s.scales[i * 3 + 2]).abs().cbrt();
        for c in cams {
            let v = c.viewmat();
            let z = v[8] * m[0] + v[9] * m[1] + v[10] * m[2] + v[11];
            if z <= 1e-4 {
                continue;
            }
            let x = v[0] * m[0] + v[1] * m[1] + v[2] * m[2] + v[3];
            let y = v[4] * m[0] + v[5] * m[1] + v[6] * m[2] + v[7];
            let (px, py) = (c.fx * x / z + c.cx, c.fy * y / z + c.cy);
            if px < 0.0 || py < 0.0 || px >= c.width as f32 || py >= c.height as f32 {
                continue;
            }
            // screen-space variance of an isotropic splat of world radius r
            let var = (r * c.fx.max(c.fy) / z).powi(2);
            comps.push(var / (var + eps2d));
        }
        if comps.is_empty() {
            continue;
        }
        comps.sort_by(f32::total_cmp);
        let c = comps[comps.len() / 2].max(1e-3);
        out.opacities[i] = (s.opacities[i] / c).min(1.0 - 1e-4);
    }
    out
}

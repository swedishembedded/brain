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

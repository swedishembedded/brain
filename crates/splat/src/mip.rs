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

/// Per-gaussian filter width in WORLD units: `scale / max_views(rate)`, the
/// rate being how many pixels per world unit a view samples the gaussian at -
/// the lens's local pixels per radian over its range, so a fisheye's
/// periphery and a pinhole's centre are both measured where they are. Zero
/// where no view saw it, which leaves it untouched.
///
/// `seen(i, v)` says whether view `v` saw gaussian `i` at all. A centre that
/// projects into a frame is not the same thing: a gaussian behind a wall
/// projects into every view of the wall, and would inherit a frequency bound
/// from cameras that never observed it. Pass `&|_, _| true` when nothing
/// better is known (a starting scene), and the frame test is all there is.
pub fn smoothing_sigma(s: &Splats, cams: &[Camera], scale: f32, seen: &(dyn Fn(usize, usize) -> bool + Sync)) -> Vec<f32> {
    let lenses: Vec<(camera::Intrinsics, [f32; 12])> = cams.iter().map(|c| (c.intrinsics(), c.viewmat())).collect();
    let best = backend_cpu::par::map_f32(s.len(), |i| {
        let m = &s.means[i * 3..i * 3 + 3];
        let mut best = 0.0f64;
        for (vi, (k, v)) in lenses.iter().enumerate() {
            if !seen(i, vi) {
                continue;
            }
            let d: [f64; 3] = std::array::from_fn(|r| (v[r * 4] * m[0] + v[r * 4 + 1] * m[1] + v[r * 4 + 2] * m[2] + v[r * 4 + 3]) as f64);
            let range = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if range <= 1e-6 {
                continue;
            }
            let Some(px) = k.project(d) else { continue };
            if px[0] < 0.0 || px[1] < 0.0 || px[0] >= k.width as f64 || px[1] >= k.height as f64 {
                continue;
            }
            if let Some(ppr) = k.pixels_per_radian(d) {
                best = best.max(ppr / range);
            }
        }
        best as f32
    });
    best.iter().map(|&f| if f > 0.0 { scale / f } else { 0.0 }).collect()
}

/// Convolve every gaussian with its own filter, preserving total energy.
///
/// Widening a gaussian spreads its mass, so opacity is scaled by
/// `sqrt(|S| / |S + sigma^2 I|)` - the same factor the 2D Mip filter uses, for
/// the same reason.
pub fn apply_3d_filter(s: &Splats, cams: &[Camera], scale: f32) -> Splats {
    let var: Vec<f32> = smoothing_sigma(s, cams, scale, &|_, _| true).iter().map(|v| v * v).collect();
    bake_filter3d(s, &var)
}

/// Bake a per-gaussian 3D filter `variance` (world units², as a fit carries
/// it) into the gaussians themselves: every axis widened by it, opacity
/// scaled by the energy factor. The result renders with no filter exactly as
/// `s` renders under it - which is what a scene handed to anything that does
/// not know about the filter (a PLY, a viewer) has to be.
pub fn bake_filter3d(s: &Splats, variance: &[f32]) -> Splats {
    let mut out = s.clone();
    for (i, &v2) in variance.iter().enumerate().take(s.len()) {
        if v2 <= 0.0 {
            continue;
        }
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
            let k = c.intrinsics();
            let d: [f64; 3] = std::array::from_fn(|q| (v[q * 4] * m[0] + v[q * 4 + 1] * m[1] + v[q * 4 + 2] * m[2] + v[q * 4 + 3]) as f64);
            let range = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let (Some(px), Some(ppr)) = (k.project(d), k.pixels_per_radian(d)) else { continue };
            if range <= 1e-4 || px[0] < 0.0 || px[1] < 0.0 || px[0] >= c.width as f64 || px[1] >= c.height as f64 {
                continue;
            }
            // screen-space variance of an isotropic splat of world radius r
            let var = (r as f64 * ppr / range).powi(2) as f32;
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

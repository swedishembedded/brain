// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Depth-driven image effects: fog and depth-of-field blur.
//!
//! Pure functions of `(rgb, depth)` — like `viz` and `stereo`, host-side and
//! testable. Each maps the depth map to a per-pixel weight and reshapes the camera
//! image by it. They compose the SAME two inputs every view mode already has (the
//! RGB frame and its depth), so the CLI just dispatches to them.

use crate::viz::Bounds;

/// Linear interpolate two bytes, `t in [0,1]`.
#[inline]
fn lerp(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t).round().clamp(0.0, 255.0) as u8
}

/// The "far-ness" of a pixel in `[0,1]` (0 = nearest, 1 = farthest), from the
/// normalized depth and the near-is-high convention.
#[inline]
fn far_of(d: f32, bounds: Bounds, near_is_high: bool) -> f32 {
    let n = bounds.norm(d);
    if near_is_high {
        1.0 - n
    } else {
        n
    }
}

/// Depth fog: blend each pixel toward `fog_color` by an amount that grows with
/// distance, so near objects stay clear and far ones dissolve into the haze
/// (`1 - exp(-density · far)`, the physical exponential-fog falloff). `density`
/// controls how quickly the fog closes in (~3–5 reads as an eerie mist).
pub fn fog(rgb: &[u8], depth: &[f32], w: u32, h: u32, bounds: Bounds, fog_color: [u8; 3], density: f32, near_is_high: bool) -> Vec<u8> {
    assert_eq!(rgb.len(), (w * h * 3) as usize, "rgb must be [h*w*3]");
    let n = (w * h) as usize;
    let mut out = vec![0u8; n * 3];
    for i in 0..n {
        let far = far_of(depth[i], bounds, near_is_high);
        let a = 1.0 - (-density * far).exp();
        for c in 0..3 {
            out[i * 3 + c] = lerp(rgb[i * 3 + c], fog_color[c], a);
        }
    }
    out
}

/// Depth-of-field blur: far pixels are blurred more than near ones, so the scene
/// appears focused up close and hazy in the distance. Built from two cumulative
/// box-blur levels blended per pixel by the depth — cheap (a few O(w·h) passes) and
/// smooth. `max_radius` is the blur radius at the far plane.
pub fn depth_blur(rgb: &[u8], depth: &[f32], w: u32, h: u32, bounds: Bounds, max_radius: u32, near_is_high: bool) -> Vec<u8> {
    assert_eq!(rgb.len(), (w * h * 3) as usize, "rgb must be [h*w*3]");
    let n = (w * h) as usize;
    if max_radius == 0 {
        return rgb.to_vec();
    }
    let r = (max_radius / 2).max(1);
    let l1 = box_blur(rgb, w, h, r); // ~r
    let l2 = box_blur(&l1, w, h, r); // ~2r (max blur)
    let mut out = vec![0u8; n * 3];
    for i in 0..n {
        let far = far_of(depth[i], bounds, near_is_high);
        // Map far in [0,1] across the three sharpness levels: sharp -> l1 -> l2.
        let (base, mix, t) = if far <= 0.5 {
            (rgb, &l1, far * 2.0)
        } else {
            (&l1[..], &l2, (far - 0.5) * 2.0)
        };
        for c in 0..3 {
            out[i * 3 + c] = lerp(base[i * 3 + c], mix[i * 3 + c], t);
        }
    }
    out
}

/// Separable box blur of radius `r` with edge-clamped prefix sums (correct at the
/// borders, O(w·h) per axis).
fn box_blur(rgb: &[u8], w: u32, h: u32, r: u32) -> Vec<u8> {
    if r == 0 {
        return rgb.to_vec();
    }
    let (wi, hi, ri) = (w as usize, h as usize, r as usize);
    // Horizontal pass.
    let mut tmp = vec![0u8; rgb.len()];
    let mut pref = vec![0i32; wi + 1];
    for y in 0..hi {
        for c in 0..3 {
            for x in 0..wi {
                pref[x + 1] = pref[x] + rgb[(y * wi + x) * 3 + c] as i32;
            }
            for x in 0..wi {
                let lo = x.saturating_sub(ri);
                let hi_ = (x + ri + 1).min(wi);
                let sum = pref[hi_] - pref[lo];
                tmp[(y * wi + x) * 3 + c] = (sum / (hi_ - lo) as i32) as u8;
            }
        }
    }
    // Vertical pass.
    let mut out = vec![0u8; rgb.len()];
    let mut pref = vec![0i32; hi + 1];
    for x in 0..wi {
        for c in 0..3 {
            for y in 0..hi {
                pref[y + 1] = pref[y] + tmp[(y * wi + x) * 3 + c] as i32;
            }
            for y in 0..hi {
                let lo = y.saturating_sub(ri);
                let hi_ = (y + ri + 1).min(hi);
                let sum = pref[hi_] - pref[lo];
                out[(y * wi + x) * 3 + c] = (sum / (hi_ - lo) as i32) as u8;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(w: u32, h: u32) -> (Vec<u8>, Vec<f32>) {
        // Left half near (depth 1.0), right half far (0.0); a red image.
        let mut rgb = vec![0u8; (w * h * 3) as usize];
        let mut depth = vec![0f32; (w * h) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                rgb[(y * w as usize + x) * 3] = 200;
                depth[y * w as usize + x] = if x < w as usize / 2 { 1.0 } else { 0.0 };
            }
        }
        (rgb, depth)
    }

    /// Fog leaves the NEAR half almost untouched and fades the FAR half toward the
    /// fog colour — the defining behaviour.
    #[test]
    fn fog_clears_near_and_hazes_far() {
        let (w, h) = (40u32, 4u32);
        let (rgb, depth) = split(w, h);
        let out = fog(&rgb, &depth, w, h, Bounds { lo: 0.0, hi: 1.0 }, [255, 255, 255], 4.0, true);
        let near = out[0]; // x=0, near, red channel
        let far = out[(w as usize - 1) * 3]; // x=w-1, far, red channel
        assert!(near > 190, "near pixel should stay ~red (200), got {near}");
        assert!(far > 240, "far pixel should be washed toward white fog, got {far}");
    }

    /// Blur leaves a NEAR edge sharp and softens a FAR edge — depth of field. Build a
    /// vertical black/white edge in each half and compare the transition sharpness.
    #[test]
    fn blur_keeps_near_sharp_and_softens_far() {
        let (w, h) = (80u32, 8u32);
        let mut rgb = vec![0u8; (w * h * 3) as usize];
        let mut depth = vec![0f32; (w * h) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                // white on the right of each quarter-edge.
                let v = if (x % 20) >= 10 { 255 } else { 0 };
                for c in 0..3 {
                    rgb[(y * w as usize + x) * 3 + c] = v;
                }
                // left half near (sharp), right half far (blurred).
                depth[y * w as usize + x] = if x < 40 { 1.0 } else { 0.0 };
            }
        }
        let out = depth_blur(&rgb, &depth, w, h, Bounds { lo: 0.0, hi: 1.0 }, 8, true);
        // A near edge (x≈10) keeps a big jump; a far edge (x≈50) is softened.
        let jump = |x: usize| (out[(x) * 3] as i32 - out[(x - 1) * 3] as i32).abs();
        let near_jump = jump(10);
        let far_jump = jump(50);
        assert!(near_jump > far_jump, "far edge must be softer than near: near {near_jump} vs far {far_jump}");
    }

    /// A flat image blurs to itself (box blur of a constant is the constant) — the
    /// sanity check that the prefix-sum math is exact at the borders.
    #[test]
    fn box_blur_of_constant_is_constant() {
        let (w, h) = (33u32, 17u32);
        let rgb = vec![137u8; (w * h * 3) as usize];
        let out = box_blur(&rgb, w, h, 5);
        assert!(out.iter().all(|&v| v == 137), "box blur of a constant must be that constant");
    }
}

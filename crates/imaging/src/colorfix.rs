// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Colour transfer between a "content" image and a "style" reference, at the
//! same resolution - what a restoration pipeline runs AFTER a generative
//! decode to pull the output's colour back onto the input's, since a
//! diffusion decode is free to drift in low-frequency colour even when the
//! high-frequency structure is right.
//!
//! Two independent methods, both host-only (pure CHW `f32` arrays, no
//! `gpu_core` dependency - the same "small enough to always run on the CPU"
//! call `crates/diffusion`'s scalar modules make):
//!
//! * [`wavelet_reconstruction`] - keep the content's HIGH frequencies, replace
//!   its LOW frequencies with the style's. This is SUPIR's own
//!   `--color_fix_type Wavelet` default: a 5-level a-trous decomposition with
//!   a fixed 3x3 low-pass kernel, doubling the dilation each level (radius
//!   `2^i`), edge-replicated at the border.
//! * [`adain`] - match the content's per-channel mean/std to the style's. A
//!   coarser, one-line-of-math match (no frequency separation), and SUPIR's
//!   other supported `--color_fix_type Adain` mode.

/// The fixed 3x3 low-pass kernel every wavelet level's blur uses, row-major.
/// The same binomial (`[1,2,1] outer [1,2,1] / 16`) approximation to a
/// Gaussian used throughout image-pyramid literature.
const KERNEL: [f32; 9] = [0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625];

/// One `a-trous` blur pass: the fixed 3x3 kernel, dilated by `radius` (so a
/// `radius = 4` pass reads 8 pixels away from centre), edge-replicated at the
/// border so the output stays defined at every pixel.
fn wavelet_blur(image: &[f32], c: usize, h: usize, w: usize, radius: i64) -> Vec<f32> {
    let (hi, wi) = (h as i64, w as i64);
    let mut out = vec![0f32; image.len()];
    for ch in 0..c {
        let plane = ch * h * w;
        for y in 0..hi {
            for x in 0..wi {
                let mut acc = 0f32;
                for ky in -1i64..=1 {
                    for kx in -1i64..=1 {
                        let sy = (y + ky * radius).clamp(0, hi - 1);
                        let sx = (x + kx * radius).clamp(0, wi - 1);
                        let k = KERNEL[((ky + 1) * 3 + (kx + 1)) as usize];
                        acc += k * image[plane + (sy as usize) * w + sx as usize];
                    }
                }
                out[plane + (y as usize) * w + x as usize] = acc;
            }
        }
    }
    out
}

/// Split `image` into its high-frequency residual and its final (level-4)
/// low-frequency band, over [`LEVELS`] a-trous levels.
const LEVELS: u32 = 5;

fn wavelet_decompose(image: &[f32], c: usize, h: usize, w: usize) -> (Vec<f32>, Vec<f32>) {
    let mut high = vec![0f32; image.len()];
    let mut cur = image.to_vec();
    for level in 0..LEVELS {
        let radius = 1i64 << level;
        let low = wavelet_blur(&cur, c, h, w, radius);
        for i in 0..high.len() {
            high[i] += cur[i] - low[i];
        }
        cur = low;
    }
    (high, cur)
}

/// `content`'s high frequencies plus `style`'s low frequencies - SUPIR's
/// `wavelet_reconstruction(content_feat, style_feat)`. Both buffers are CHW
/// `f32`, same `(c, h, w)`, any value range (this is a linear operation, so it
/// commutes with any shared affine rescale the caller applies before or
/// after).
pub fn wavelet_reconstruction(content: &[f32], style: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    assert_eq!(content.len(), c * h * w, "wavelet_reconstruction: content size");
    assert_eq!(style.len(), c * h * w, "wavelet_reconstruction: style size");
    let (content_high, _) = wavelet_decompose(content, c, h, w);
    let (_, style_low) = wavelet_decompose(style, c, h, w);
    content_high.iter().zip(&style_low).map(|(&hi, &lo)| hi + lo).collect()
}

/// Per-channel mean and (population) standard deviation, floored at a small
/// epsilon so a flat (zero-variance) channel does not divide by zero.
fn channel_stats(image: &[f32], c: usize, hw: usize) -> (Vec<f32>, Vec<f32>) {
    let mut mean = vec![0f32; c];
    let mut std = vec![0f32; c];
    for ch in 0..c {
        let plane = &image[ch * hw..(ch + 1) * hw];
        let m = plane.iter().sum::<f32>() / hw as f32;
        let var = plane.iter().map(|&v| (v - m) * (v - m)).sum::<f32>() / hw as f32;
        mean[ch] = m;
        std[ch] = var.sqrt().max(1e-5);
    }
    (mean, std)
}

/// Adaptive Instance Normalization colour transfer: rescale each of
/// `content`'s channels to `style`'s per-channel mean/std. SUPIR's
/// `--color_fix_type Adain`.
pub fn adain(content: &[f32], style: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let hw = h * w;
    assert_eq!(content.len(), c * hw, "adain: content size");
    assert_eq!(style.len(), c * hw, "adain: style size");
    let (cm, cs) = channel_stats(content, c, hw);
    let (sm, ss) = channel_stats(style, c, hw);
    let mut out = vec![0f32; content.len()];
    for ch in 0..c {
        let (m0, s0, m1, s1) = (cm[ch], cs[ch], sm[ch], ss[ch]);
        for i in 0..hw {
            let idx = ch * hw + i;
            out[idx] = (content[idx] - m0) / s0 * s1 + m1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat image is its own low-frequency band at every level (the a-trous
    /// blur of a constant image is that constant, at any radius), so its high
    /// frequency is exactly zero and `wavelet_reconstruction` against ANY
    /// style collapses to the style's own flat value.
    #[test]
    fn wavelet_of_a_flat_image_has_zero_high_frequency() {
        let (c, h, w) = (2usize, 9usize, 9usize);
        let flat = vec![0.3f32; c * h * w];
        let (high, low) = wavelet_decompose(&flat, c, h, w);
        assert!(high.iter().all(|&v| v.abs() < 1e-6), "flat image must have ~zero high-frequency content");
        assert!(low.iter().all(|&v| (v - 0.3).abs() < 1e-6));
    }

    /// Reconstructing a flat content against a flat style of a different
    /// value must land exactly on the style's value: high(content) = 0, so
    /// the result is purely `low(style)`.
    #[test]
    fn wavelet_reconstruction_of_two_flat_images_takes_the_style_value() {
        let (c, h, w) = (1usize, 9usize, 9usize);
        let content = vec![0.1f32; c * h * w];
        let style = vec![0.9f32; c * h * w];
        let out = wavelet_reconstruction(&content, &style, c, h, w);
        assert!(out.iter().all(|&v| (v - 0.9).abs() < 1e-5), "{out:?}");
    }

    /// A non-flat content's fine structure (its edges) must survive - the
    /// whole point of the method over a plain style copy.
    #[test]
    fn wavelet_reconstruction_preserves_content_structure() {
        let (c, h, w) = (1usize, 16usize, 16usize);
        let mut content = vec![0f32; c * h * w];
        for y in 0..h {
            for x in 0..w {
                content[y * w + x] = if (x + y) % 2 == 0 { 0.0 } else { 1.0 };
            }
        }
        let style = vec![0.5f32; c * h * w]; // flat style: low_freq(style) == 0.5 everywhere
        let out = wavelet_reconstruction(&content, &style, c, h, w);
        // The checkerboard's high-frequency swing must still be visible: the
        // reconstructed values must differ from each other (not collapsed to
        // one flat value the way a plain colour copy would).
        let lo = out.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(hi - lo > 0.5, "checkerboard structure was lost: min {lo} max {hi}");
    }

    #[test]
    fn adain_matches_content_stats_to_style_stats() {
        let (c, h, w) = (1usize, 4usize, 4usize);
        let content: Vec<f32> = (0..16).map(|i| i as f32).collect(); // mean 7.5
        let style = vec![10.0f32; 16]; // mean 10, std 0 -> floored epsilon
        let out = adain(&content, &style, c, h, w);
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        assert!((mean - 10.0).abs() < 1e-2, "{mean}");
    }

    #[test]
    fn adain_is_a_no_op_when_content_already_matches_style_stats() {
        let (c, h, w) = (1usize, 4usize, 4usize);
        let content: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let out = adain(&content, &content, c, h, w);
        for (a, b) in out.iter().zip(&content) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }
}

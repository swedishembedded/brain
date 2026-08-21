// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Cheap, objective temporal-stability statistics for a decoded clip - the
//! frame-to-frame difference curve, and the "does it blow up somewhere"
//! summary of it.
//!
//! Swedish Embedded AB implements objective quality gates for generative
//! video pipelines for its clients. If your team needs a regression signal
//! that catches a clip degrading part-way through without a human watching
//! it, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # Why this metric
//!
//! A generative video defect that a cosine-similarity gate cannot see is a
//! clip that is *locally* fine and *globally* incoherent: every frame is a
//! plausible image, the statistics are in range, nothing is non-finite, and
//! the video still falls apart half way through. What that looks like
//! numerically is the mean absolute difference between CONSECUTIVE frames
//! jumping by an order of magnitude at one point and staying there.
//!
//! [`frame_to_frame_diffs`] is that curve. It downsamples each frame to a
//! fixed [`PROBE`]x[`PROBE`] box average first, which is what makes it
//! resolution-independent (so a 720p run and a 1080p run produce comparable
//! numbers) and insensitive to the high-frequency detail that legitimately
//! differs between resolutions. [`blowup_ratio`] reduces the curve to the one
//! number a gate wants: the largest single value over the MEDIAN, which is
//! ~1-3x for a clip with normal motion however fast that motion is, and an
//! order of magnitude for a clip that disintegrates.
//!
//! Both are deliberately blind to *content*: a clip can score perfectly and
//! be the wrong video. This is a stability gate, not a quality one.

/// The side of the box-downsampled probe grid each frame is reduced to before
/// differencing. Small enough that the metric is about motion and structure
/// rather than sensor-scale detail, large enough that a localized
//  disintegration cannot average away.
pub const PROBE: usize = 128;

/// Box-average one `[3, h, w]`-plane-major frame down to `[3, PROBE, PROBE]`,
/// rescaled from the decoder's `[-1, 1]` to a `[0, 255]` display range so the
/// numbers read like pixel differences.
fn probe_frame(pixels: &[f32], frames: usize, fi: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0f32; 3 * PROBE * PROBE];
    for c in 0..3 {
        let plane = &pixels[(c * frames + fi) * h * w..(c * frames + fi + 1) * h * w];
        for py in 0..PROBE {
            let (y0, y1) = (py * h / PROBE, (((py + 1) * h).div_ceil(PROBE)).min(h));
            let y1 = y1.max(y0 + 1);
            for px in 0..PROBE {
                let (x0, x1) = (px * w / PROBE, (((px + 1) * w).div_ceil(PROBE)).min(w));
                let x1 = x1.max(x0 + 1);
                let mut acc = 0f64;
                for y in y0..y1 {
                    for x in x0..x1 {
                        acc += plane[y * w + x] as f64;
                    }
                }
                let mean = acc / ((y1 - y0) * (x1 - x0)) as f64;
                out[(c * PROBE + py) * PROBE + px] = ((mean + 1.0) * 127.5) as f32;
            }
        }
    }
    out
}

/// Mean absolute difference between consecutive frames of a decoded
/// `[3, frames, h, w]` clip, on a [`PROBE`]x[`PROBE`] box downsample, in
/// `[0, 255]` display units.
///
/// Returns `frames - 1` values; entry `i` is the difference between frame
/// `i + 1` and frame `i`, so it indexes the way a "the jump is at frame 18"
/// report does.
pub fn frame_to_frame_diffs(pixels: &[f32], frames: usize, h: usize, w: usize) -> Vec<f32> {
    assert_eq!(pixels.len(), 3 * frames * h * w, "frame_to_frame_diffs: expected [3, {frames}, {h}, {w}]");
    if frames < 2 {
        return Vec::new();
    }
    let mut prev = probe_frame(pixels, frames, 0, h, w);
    let mut out = Vec::with_capacity(frames - 1);
    for fi in 1..frames {
        let cur = probe_frame(pixels, frames, fi, h, w);
        let sum: f64 = cur.iter().zip(&prev).map(|(a, b)| (a - b).abs() as f64).sum();
        out.push((sum / cur.len() as f64) as f32);
        prev = cur;
    }
    out
}

/// The largest frame-to-frame difference over the MEDIAN one.
///
/// A clip with normal motion, fast or slow, holds this near 1 because the
/// median tracks whatever that clip's own motion happens to be. A clip that
/// disintegrates part-way through pushes it into double digits, which is why
/// this - rather than any absolute difference threshold - is the number a
/// resolution-independent gate can bound.
///
/// Returns `1.0` for a clip too short to have two differences, and for one
/// whose median difference is zero (a static clip, where the ratio is not
/// defined and nothing is unstable).
pub fn blowup_ratio(diffs: &[f32]) -> f32 {
    if diffs.len() < 2 {
        return 1.0;
    }
    let mut s: Vec<f32> = diffs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = s[s.len() / 2];
    let max = *s.last().unwrap_or(&0.0);
    if median <= 0.0 {
        return 1.0;
    }
    max / median
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `[3, frames, h, w]` clip whose frame `fi` is the constant `f(fi)`.
    fn constant_frames(vals: &[f32], h: usize, w: usize) -> Vec<f32> {
        let frames = vals.len();
        let mut v = vec![0f32; 3 * frames * h * w];
        for c in 0..3 {
            for (fi, &x) in vals.iter().enumerate() {
                let base = (c * frames + fi) * h * w;
                v[base..base + h * w].fill(x);
            }
        }
        v
    }

    #[test]
    fn a_steady_pan_scores_flat_and_a_late_blowup_does_not() {
        let (h, w) = (64usize, 48usize);
        // Steady: each frame 0.02 brighter than the last -> 0.02*127.5 = 2.55.
        let steady: Vec<f32> = (0..12).map(|i| -0.5 + 0.02 * i as f32).collect();
        let d = frame_to_frame_diffs(&constant_frames(&steady, h, w), steady.len(), h, w);
        for x in &d {
            assert!((x - 2.55).abs() < 1e-2, "steady clip should difference at 2.55, got {x}");
        }
        assert!(blowup_ratio(&d) < 1.05, "a steady clip must not read as a blowup: {}", blowup_ratio(&d));

        // The same clip with the last three frames thrashing.
        let mut broken = steady.clone();
        broken[9] = 0.6;
        broken[10] = -0.6;
        broken[11] = 0.6;
        let d2 = frame_to_frame_diffs(&constant_frames(&broken, h, w), broken.len(), h, w);
        assert!(blowup_ratio(&d2) > 10.0, "a late blowup must read as one: {}", blowup_ratio(&d2));
    }

    #[test]
    fn the_metric_is_resolution_independent_for_the_same_content() {
        let vals: Vec<f32> = (0..6).map(|i| -0.3 + 0.05 * i as f32).collect();
        let a = frame_to_frame_diffs(&constant_frames(&vals, 64, 64), vals.len(), 64, 64);
        let b = frame_to_frame_diffs(&constant_frames(&vals, 512, 288), vals.len(), 512, 288);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-3, "same content at two resolutions must score the same: {x} vs {y}");
        }
    }

    #[test]
    fn a_static_clip_is_not_a_blowup() {
        let vals = vec![0.1f32; 8];
        let d = frame_to_frame_diffs(&constant_frames(&vals, 32, 32), vals.len(), 32, 32);
        assert_eq!(blowup_ratio(&d), 1.0, "a zero-motion clip has no defined ratio and must not read as unstable");
    }
}

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Keypoint detection at the scales photogrammetry needs: fine texture
//! (gravel, bark, print) is where most of a capture's correspondences come
//! from, and every camera is only as precise as the keypoints it was solved
//! from.
//!
//! Swedish Embedded AB implements feature detection and matching for
//! photogrammetry. If your team needs photographs turned into calibrated
//! cameras, you can procure our services by sending an email to
//! info@swedishembedded.com.

use sfm::sift::{detect, SiftCfg};

/// A grid of bright gaussian dots of `sigma` px, centres at odd quarter
/// pixels so no dot sits on the sampling grid, over a dark field.
fn dots(w: usize, h: usize, sigma: f32, pitch: usize) -> (Vec<f32>, Vec<(f32, f32)>) {
    let mut centres = Vec::new();
    for gy in 1..h / pitch {
        for gx in 1..w / pitch {
            centres.push(((gx * pitch) as f32 + 0.25, (gy * pitch) as f32 + 0.75));
        }
    }
    let mut img = vec![0.1f32; w * h];
    for &(cx, cy) in &centres {
        let r = (4.0 * sigma).ceil() as isize;
        for dy in -r..=r {
            for dx in -r..=r {
                let (x, y) = (cx as isize + dx, cy as isize + dy);
                if x < 0 || y < 0 || x >= w as isize || y >= h as isize {
                    continue;
                }
                // continuous coordinates: pixel i's centre is i + 0.5
                let (ex, ey) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
                img[y as usize * w + x as usize] += 0.8 * (-(ex * ex + ey * ey) / (2.0 * sigma * sigma)).exp();
            }
        }
    }
    (img, centres)
}

/// Dots finer than the first octave's scale are found, where they are. A
/// dot of sigma `b` peaks in the difference of gaussians at about `0.9 b`,
/// and the lowest level extrema are sought at is `σ0 · 2^(1/3) = 2.0`, so an
/// undoubled pyramid misses everything below `b ≈ 2.2` px. Lowe (IJCV 2004,
/// §3.3) doubles the input for exactly this - it multiplies the stable
/// keypoints by almost four - and so do the SfM systems that follow him.
#[test]
fn fine_texture_is_detected_where_it_is() {
    let (w, h) = (192, 160);
    let (img, centres) = dots(w, h, 1.2, 12);
    let (kps, _) = detect(&img, w, h, &SiftCfg::default());
    let mut found = 0;
    let mut worst = 0.0f32;
    for &(cx, cy) in &centres {
        let near = kps
            .iter()
            .map(|k| ((k.x - cx).powi(2) + (k.y - cy).powi(2)).sqrt())
            .fold(f32::INFINITY, f32::min);
        if near < 1.0 {
            found += 1;
            worst = worst.max(near);
        }
    }
    assert!(
        found == centres.len(),
        "{found} of {} sigma-1.2 dots detected; fine texture has to produce keypoints",
        centres.len()
    );
    assert!(worst < 0.1, "a dot's keypoint is {worst:.3} px from its centre; the pyramid misplaces fine features");
}

/// The upsampled octave must not shift the coarse ones: a large dot is found
/// at its centre too.
#[test]
fn coarse_features_stay_where_they_are() {
    let (w, h) = (256, 256);
    let (img, centres) = dots(w, h, 5.0, 64);
    let (kps, _) = detect(&img, w, h, &SiftCfg::default());
    for &(cx, cy) in &centres {
        let near = kps
            .iter()
            .filter(|k| k.sigma > 3.0)
            .map(|k| ((k.x - cx).powi(2) + (k.y - cy).powi(2)).sqrt())
            .fold(f32::INFINITY, f32::min);
        assert!(near < 0.1, "the sigma-5 dot at ({cx}, {cy}) is found {near:.3} px away");
    }
}


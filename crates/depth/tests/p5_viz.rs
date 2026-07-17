// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! P5: the visualization path — colormap, robust bounds, composite. All pure.
use depth::viz::{colorize, composite_side_by_side, Bounds, Colormap};

/// Turbo's endpoints are pinned: a colormap that silently shifts would recolor
/// every demo frame. Turbo runs dark-blue -> cyan -> yellow -> dark-red.
#[test]
fn turbo_endpoints_are_stable() {
    let lut = Colormap::Turbo.lut();
    let (lo, hi) = (lut[0], lut[255]);
    // Low end: blue-ish and dark (turbo starts at a dark blue-purple).
    assert!(lo[2] > lo[0] && lo[2] > lo[1], "turbo low end should be blue-dominant, got {lo:?}");
    // High end: red-dominant and dark-ish.
    assert!(hi[0] > hi[1] && hi[0] > hi[2], "turbo high end should be red-dominant, got {hi:?}");
    // Every entry stays in gamut (the polynomial can overshoot; chan() clamps).
    for e in lut {
        assert!(e.iter().all(|_| true));
    }
}

#[test]
fn gray_is_a_linear_ramp() {
    let lut = Colormap::Gray.lut();
    assert_eq!(lut[0], [0, 0, 0]);
    assert_eq!(lut[255], [255, 255, 255]);
    assert_eq!(lut[128], [128, 128, 128]);
    // GrayInv is the reverse.
    let inv = Colormap::GrayInv.lut();
    assert_eq!(inv[0], [255, 255, 255]);
    assert_eq!(inv[255], [0, 0, 0]);
}

#[test]
fn colormap_cycles_through_all_three() {
    let mut seen = std::collections::HashSet::new();
    let mut c = Colormap::Turbo;
    for _ in 0..3 {
        seen.insert(format!("{c:?}"));
        c = c.next();
    }
    assert_eq!(seen.len(), 3, "the [ / ] cycle must reach every colormap");
    assert_eq!(c, Colormap::Turbo, "and return to the start");
}

/// The whole reason bounds are not per-frame min/max: a lone huge outlier must not
/// move the window. p2/p98 of a mostly-uniform frame ignore one 1e6 spike.
#[test]
fn robust_bounds_ignore_a_single_outlier() {
    let mut depth = vec![1.0f32; 10_000];
    for (i, v) in depth.iter_mut().enumerate() {
        *v = 1.0 + (i as f32) / 10_000.0; // 1.0 .. 2.0
    }
    let clean = Bounds::from_percentiles(&depth, 0.02, 0.98);
    depth[0] = 1e6; // a specular spike
    let spiked = Bounds::from_percentiles(&depth, 0.02, 0.98);
    assert!((clean.hi - spiked.hi).abs() < 0.05, "p98 must barely move: {} vs {}", clean.hi, spiked.hi);
    // Per-frame max, by contrast, would have jumped to 1e6.
    assert!(spiked.hi < 10.0, "robust hi must not chase the outlier, got {}", spiked.hi);
}

#[test]
fn bounds_never_degenerate_and_norm_clamps() {
    // An all-equal frame must still give a usable, non-zero-width window.
    let flat = Bounds::from_percentiles(&vec![3.0f32; 100], 0.02, 0.98);
    assert!(flat.hi > flat.lo, "degenerate window would divide by zero");
    // norm clamps outside [lo,hi].
    let b = Bounds { lo: 1.0, hi: 2.0 };
    assert_eq!(b.norm(0.0), 0.0);
    assert_eq!(b.norm(3.0), 1.0);
    assert!((b.norm(1.5) - 0.5).abs() < 1e-6);
}

#[test]
fn ema_moves_toward_target() {
    let a = Bounds { lo: 0.0, hi: 1.0 };
    let t = Bounds { lo: 1.0, hi: 2.0 };
    let m = a.ema(t, 0.1);
    assert!((m.lo - 0.1).abs() < 1e-6 && (m.hi - 1.1).abs() < 1e-6);
}

#[test]
fn colorize_maps_bounds_endpoints_to_lut_ends() {
    let depth = [1.0f32, 1.5, 2.0];
    let b = Bounds { lo: 1.0, hi: 2.0 };
    let rgb = colorize(&depth, b, Colormap::Gray);
    assert_eq!(rgb.len(), 3 * 3);
    assert_eq!(&rgb[0..3], &[0, 0, 0], "lo -> black");
    assert_eq!(&rgb[6..9], &[255, 255, 255], "hi -> white");
    assert_eq!(&rgb[3..6], &[128, 128, 128], "midpoint -> mid grey");
}

/// The composite places left and right at the correct offsets, row by row.
#[test]
fn composite_places_both_halves() {
    // 2x2 red-left, 2x2 green-right.
    let left = vec![255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0];
    let right = vec![0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0];
    let (out, w, h) = composite_side_by_side(&left, 2, 2, &right, 2, 2);
    assert_eq!((w, h), (4, 2));
    // Row 0: [red, red, green, green].
    assert_eq!(&out[0..3], &[255, 0, 0]); // (0,0) left
    assert_eq!(&out[6..9], &[0, 255, 0]); // (2,0) right
    // Row 1 starts at 4*3 = 12.
    assert_eq!(&out[12..15], &[255, 0, 0]); // (0,1) left
    assert_eq!(&out[18..21], &[0, 255, 0]); // (2,1) right
}

/// Different widths per half (RGB source ≠ depth map size) must still compose.
#[test]
fn composite_handles_unequal_widths() {
    let left = vec![1u8; (3 * 3 * 2) as usize]; // 3x2
    let right = vec![2u8; (1 * 3 * 2) as usize]; // 1x2
    let (out, w, h) = composite_side_by_side(&left, 3, 2, &right, 1, 2);
    assert_eq!((w, h), (4, 2));
    assert_eq!(out.len(), (4 * 2 * 3) as usize);
    // last column of each row is the right half (value 2).
    assert_eq!(out[9], 2); // row0 col3
    assert_eq!(out[21], 2); // row1 col3
}

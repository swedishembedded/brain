// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Calibration must preprocess the way inference does.
//!
//! `brain depth calib` used to letterbox every image into a padded square with a
//! 0.5 grey fill, while `Predictor::begin` does an aspect-preserving bilinear
//! resize to a multiple of 32 with no pad. So the INT8 activation ranges were
//! fitted to a resampler, a geometry and a border the model never sees — and
//! `zipdepth::predict`'s own module docs record that letterboxing was REMOVED
//! because it visibly degrades the depth.
//!
//! These tests pin the property that fixes it: one preprocessing function, used
//! by both paths, that does not letterbox. They need no checkpoint, because the
//! defect was in the transform rather than in the weights.
//!
//! NOT covered here: how far the INT8 scales actually move. That needs a
//! ZipDepth checkpoint and a calibration set, neither of which is on this box.

use zipdepth::predict::{preprocess_chw, target_size};

/// A non-square image must stay non-square. A letterbox would return `input²`.
#[test]
fn preprocessing_preserves_aspect_and_does_not_pad() {
    let (w0, h0, input) = (640u32, 480u32, 384u32);
    let hwc = vec![0.25f32; (w0 * h0 * 3) as usize];

    let (chw, th, tw) = preprocess_chw(&hwc, w0, h0, input);

    assert_ne!(th, tw, "a 4:3 image must not become square — that is the letterbox bug");
    assert_eq!((th, tw), target_size(w0, h0, input), "must agree with the predictor's own sizing");
    assert_eq!(th.min(tw), input, "the SHORTER side is the model input");
    assert_eq!(th % 32, 0, "height must be a multiple of 32");
    assert_eq!(tw % 32, 0, "width must be a multiple of 32");
    assert_eq!(chw.len(), (3 * th * tw) as usize);
}

/// A constant image must come back constant. A padded letterbox would introduce
/// the 0.5 grey border — the value the model never sees at inference.
#[test]
fn no_fill_value_is_introduced() {
    let (w0, h0, input) = (800u32, 500u32, 384u32);
    let hwc = vec![0.25f32; (w0 * h0 * 3) as usize];

    let (chw, _, _) = preprocess_chw(&hwc, w0, h0, input);

    let lo = chw.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = chw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!((lo - 0.25).abs() < 1e-5 && (hi - 0.25).abs() < 1e-5, "resize introduced values outside the input: [{lo}, {hi}]");
}

/// Portrait and landscape must produce transposed shapes. A square letterbox
/// erases the distinction, which is why calibration saw one shape for
/// everything.
#[test]
fn portrait_and_landscape_are_distinguishable() {
    let input = 384u32;
    let land = preprocess_chw(&vec![0.5f32; (640 * 480 * 3) as usize], 640, 480, input);
    let port = preprocess_chw(&vec![0.5f32; (480 * 640 * 3) as usize], 480, 640, input);
    assert_eq!((land.1, land.2), (port.2, port.1), "landscape and portrait should transpose");
    assert_ne!((land.1, land.2), (port.1, port.2), "they must not collapse to one shape");
}

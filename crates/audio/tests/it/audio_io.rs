// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! wav round-trip + mel front-end smoke tests (pure CPU, no GPU).

use audio::mel::{log_mel, MelConfig};
use audio::{resample_linear, wav};

#[test]
fn wav_roundtrip_16bit() {
    let sr = 24000;
    let n = 2000;
    let samples: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).sin() * 0.7).collect();
    let bytes = wav::encode(&samples, sr);
    let w = wav::parse(&bytes).expect("parse");
    assert_eq!(w.sample_rate, sr);
    assert_eq!(w.samples.len(), n);
    // 16-bit quantization error bound.
    let max_err = samples.iter().zip(&w.samples).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    // 16-bit LSB ~3.05e-5; allow ~2 LSB for the 32767/32768 encode/decode asymmetry.
    assert!(max_err < 1e-4, "roundtrip err {max_err}");
}

#[test]
fn resample_changes_length() {
    let s: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.1).sin()).collect();
    let up = resample_linear(&s, 16000, 24000);
    assert!((up.len() as i64 - 1500).abs() <= 2, "got {}", up.len());
}

#[test]
fn log_mel_shapes_and_finite() {
    let cfg = MelConfig::default_24k();
    let s: Vec<f32> = (0..24000).map(|i| (i as f32 * 0.02).sin() * 0.5).collect();
    let (m, frames) = log_mel(&s, &cfg);
    assert!(frames > 80, "frames {frames}");
    assert_eq!(m.len(), frames * cfg.n_mels);
    assert!(m.iter().all(|v| v.is_finite()), "non-finite mel");
}

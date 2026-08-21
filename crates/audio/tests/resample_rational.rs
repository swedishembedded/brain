// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Correctness gate for `audio::resample::rational`, the Kaiser-windowed-sinc
//! polyphase resampler:
//!   1. a below-cutoff tone survives resampling near-unchanged (passband);
//!   2. an above-new-Nyquist tone is suppressed, not aliased down into the
//!      output band (`resample_linear` cannot do this - it has no filter at
//!      all, so this is exactly the case it fails);
//!   3. up-then-down round-trips back to the original away from the edges.

use audio::resample::rational;

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|&v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

fn tone(f: f32, sr: u32, n: usize) -> Vec<f32> {
    (0..n).map(|i| (2.0 * std::f32::consts::PI * f * i as f32 / sr as f32).sin()).collect()
}

/// A 3 kHz tone at 24 kHz is well inside 16 kHz's Nyquist (8 kHz) - resampling
/// 24k -> 16k should preserve it at close to unit amplitude.
#[test]
fn passband_tone_survives_downsample() {
    let x = tone(3000.0, 24000, 4800);
    let y = rational(&x, 16000, 24000);
    // exact ratio 2:3 -> output length 3200
    assert_eq!(y.len(), 3200);
    // skip filter-edge frames on both sides
    let edge = 200;
    let interior = &y[edge..y.len() - edge];
    let r = rms(interior);
    assert!(r > 0.6, "passband tone attenuated too much: rms={r}");
}

/// A 10 kHz tone at 24 kHz sits ABOVE 16 kHz's new Nyquist (8 kHz). A correct
/// resampler's anti-alias filter must suppress it, not fold it down into the
/// audible band the way naive decimation (or `resample_linear`) would.
#[test]
fn above_nyquist_tone_is_suppressed_not_aliased() {
    let x = tone(10000.0, 24000, 4800);
    let y = rational(&x, 16000, 24000);
    let edge = 200;
    let interior = &y[edge..y.len() - edge];
    let r = rms(interior);
    // input tone has rms ~0.707; a properly filtered result should be far
    // below that, not merely "somewhat smaller".
    assert!(r < 0.15, "above-Nyquist energy leaked through: rms={r}");
}

/// Resampling up then back down by the inverse ratio should approximately
/// recover the original signal away from the filter's edge transients.
#[test]
fn up_then_down_roundtrips() {
    let x = tone(1200.0, 16000, 3200);
    let up = rational(&x, 24000, 16000); // 16k -> 24k (up=3,down=2)
    let back = rational(&up, 16000, 24000); // 24k -> 16k (up=2,down=3)
    assert_eq!(back.len(), x.len());
    let edge = 300;
    let lo = edge;
    let hi = x.len() - edge;
    assert!(hi > lo, "signal too short for an interior comparison");
    let max_abs = x[lo..hi].iter().zip(&back[lo..hi]).map(|(a, b): (&f32, &f32)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_abs < 0.1, "up-then-down roundtrip interior maxdiff {max_abs}");
}

/// Ratio 1:1 (or any up==down after gcd reduction) is a no-op.
#[test]
fn unity_ratio_is_identity() {
    let x = tone(500.0, 8000, 100);
    let y = rational(&x, 2, 2);
    assert_eq!(x, y);
}

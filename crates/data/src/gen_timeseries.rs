// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Synthetic 3-phase time-series dataset generator.
//!
//! Ported from nanogpt's `data_generators/timeseries.py`. Synthesizes a
//! balanced 3-phase electrical signal of shape `(n_steps, 3)`: three sinusoids
//! 120 degrees apart, each built from a fundamental ([`FUNDAMENTAL_FREQ`]) plus
//! 3rd/5th harmonics, slow amplitude modulation, and a small Gaussian noise
//! term. The output is row-major flattened (`n_steps * 3`), so row `t` is
//! `[phaseA, phaseB, phaseC]`.
//!
//! We cannot byte-reproduce numpy's RNG stream, so this is *functionally*
//! equivalent to the reference (same structure, parameters, and distributions)
//! rather than bit-identical. For a fixed `seed` the output is deterministic.

use crate::rng::Rng;

/// Number of features per time step (Phase A, B, C).
pub const N_FEATURES: usize = 3;

/// Fundamental frequency in cycles per time step (about 50 samples/cycle).
const FUNDAMENTAL_FREQ: f64 = 0.02;
/// Base amplitude of the fundamental component.
const AMPLITUDE: f64 = 1.0;
/// Standard deviation of the additive Gaussian noise.
const NOISE_LEVEL: f64 = 0.03;
/// Frequency of the slow amplitude modulation envelope.
const AMPLITUDE_MODULATION_FREQ: f64 = 0.0005;
/// Depth of the amplitude modulation (0 to 1).
const AMPLITUDE_MODULATION_DEPTH: f64 = 0.15;
/// Harmonics added for electrical realism: `(order, amplitude)`.
const HARMONICS: [(f64, f64); 2] = [(3.0, 0.1), (5.0, 0.05)];

/// Generate the 3-phase signal: returns `n_steps * 3` f32 values, row-major
/// (row t = [phaseA, phaseB, phaseC]). Deterministic for a fixed `seed`.
pub fn generate(n_steps: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);

    // Phase shifts for a balanced 3-phase system (120 degrees apart).
    let phase_shifts = [
        0.0,
        2.0 * std::f64::consts::PI / 3.0,
        4.0 * std::f64::consts::PI / 3.0,
    ];

    let omega = 2.0 * std::f64::consts::PI * FUNDAMENTAL_FREQ;
    let omega_mod = 2.0 * std::f64::consts::PI * AMPLITUDE_MODULATION_FREQ;

    let mut out = vec![0.0f32; n_steps * N_FEATURES];

    // Mirror the reference order: iterate phase-major, drawing a full noise
    // vector per phase. This keeps the noise stream usage identical in shape to
    // numpy's `randn(n_steps)` per phase.
    for (i, &shift) in phase_shifts.iter().enumerate() {
        for t in 0..n_steps {
            let tf = t as f64;

            // Fundamental component.
            let mut value = AMPLITUDE * (omega * tf - shift).sin();

            // Harmonics.
            for &(order, h_amp) in &HARMONICS {
                value += h_amp * (order * omega * tf - shift).sin();
            }

            // Amplitude modulation.
            let modulation = 1.0 + AMPLITUDE_MODULATION_DEPTH * (omega_mod * tf).sin();
            value *= modulation;

            // Gaussian noise.
            value += rng.next_gaussian() * NOISE_LEVEL;

            out[t * N_FEATURES + i] = value as f32;
        }
    }

    out
}

/// Chronological train/val split with a temporal gap (= `context_length`) to
/// prevent leakage. Input is row-major `n_steps*3`. Returns (train, val), each
/// row-major `*3`. Mirrors `save_dataset`'s split logic.
pub fn split(data: &[f32], train_split: f64, context_length: usize) -> (Vec<f32>, Vec<f32>) {
    let n_steps = data.len() / N_FEATURES;

    // Use context_length as the temporal gap (stricter than forecast_horizon),
    // so validation inputs never immediately follow training-label timestamps.
    let gap = context_length;
    let split_idx = (n_steps as f64 * train_split).floor() as usize;
    let val_start_idx = (split_idx + gap).min(n_steps);

    let train = data[..split_idx * N_FEATURES].to_vec();
    let val = data[val_start_idx * N_FEATURES..].to_vec();

    (train, val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_length_matches_shape() {
        let n_steps = 1000;
        let data = generate(n_steps, 42);
        assert_eq!(data.len(), n_steps * N_FEATURES);
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let a = generate(2000, 42);
        let b = generate(2000, 42);
        assert_eq!(a, b);

        // A different seed should (almost certainly) change the noise stream.
        let c = generate(2000, 43);
        assert_ne!(a, c);
    }

    #[test]
    fn balanced_three_phase_sums_to_zero_on_average() {
        let n_steps = 20_000;
        let data = generate(n_steps, 42);

        let mut sum_of_row_sums = 0.0f64;
        for t in 0..n_steps {
            let row = &data[t * N_FEATURES..t * N_FEATURES + N_FEATURES];
            sum_of_row_sums += (row[0] + row[1] + row[2]) as f64;
        }
        let mean = sum_of_row_sums / n_steps as f64;

        // A balanced 3-phase system cancels; only noise/harmonic residue remains.
        assert!(mean.abs() < 0.05, "mean row-sum too large: {mean}");
    }

    #[test]
    fn split_sizes_and_offsets_are_correct() {
        let n_steps = 10_000;
        let train_split = 0.9;
        let context_length = 60;

        let data = generate(n_steps, 42);
        let (train, val) = split(&data, train_split, context_length);

        let split_idx = (n_steps as f64 * train_split).floor() as usize;
        let gap = context_length;
        let val_start_idx = split_idx + gap;

        // Train covers [0, split_idx).
        assert_eq!(train.len(), split_idx * N_FEATURES);

        // Val covers [split_idx + gap, n_steps).
        assert_eq!(val.len(), (n_steps - val_start_idx) * N_FEATURES);

        // The gap equals context_length and is excluded from both splits.
        let train_steps = train.len() / N_FEATURES;
        let val_steps = val.len() / N_FEATURES;
        assert_eq!(n_steps - train_steps - val_steps, gap);

        // Val data begins exactly at the post-gap offset in the source array.
        assert_eq!(val[0], data[val_start_idx * N_FEATURES]);
    }
}

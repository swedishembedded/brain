// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A tiny deterministic PRNG for scenario generation — seeded, reproducible,
//! dependency-free. SplitMix64 for the stream, Box–Muller for Gaussians.
//!
//! Reproducibility matters here: a scenario sample must be identical on every
//! machine and run so "data the model has never seen" is *guaranteed unseen* yet
//! exactly regenerable for scoring.

/// SplitMix64 PRNG.
pub struct Rng {
    state: u64,
    spare: Option<f32>,
}

impl Rng {
    /// Seed the stream.
    pub fn new(seed: u64) -> Rng {
        Rng { state: seed ^ 0x9E37_79B9_7F4A_7C15, spare: None }
    }

    /// Next raw u64.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / (1u64 << 24) as f32
    }

    /// Standard normal via Box–Muller (caches the spare deviate).
    pub fn normal(&mut self) -> f32 {
        if let Some(s) = self.spare.take() {
            return s;
        }
        // avoid u=0 for the log
        let u1 = (self.uniform()).max(1e-7);
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        self.spare = Some(r * theta.sin());
        r * theta.cos()
    }

    /// Normal with given mean and std.
    pub fn normal_with(&mut self, mean: f32, std: f32) -> f32 {
        mean + std * self.normal()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_for_a_seed() {
        let a: Vec<u64> = { let mut r = Rng::new(42); (0..5).map(|_| r.next_u64()).collect() };
        let b: Vec<u64> = { let mut r = Rng::new(42); (0..5).map(|_| r.next_u64()).collect() };
        assert_eq!(a, b);
        let c: Vec<u64> = { let mut r = Rng::new(43); (0..5).map(|_| r.next_u64()).collect() };
        assert_ne!(a, c);
    }

    #[test]
    fn normal_is_roughly_standard() {
        let mut r = Rng::new(7);
        let n = 20000;
        let xs: Vec<f32> = (0..n).map(|_| r.normal()).collect();
        let mean = xs.iter().sum::<f32>() / n as f32;
        let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
        assert!(mean.abs() < 0.05, "mean {mean}");
        assert!((var - 1.0).abs() < 0.1, "var {var}");
    }

    #[test]
    fn uniform_in_range() {
        let mut r = Rng::new(1);
        for _ in 0..10000 {
            let u = r.uniform();
            assert!((0.0..1.0).contains(&u));
        }
    }
}

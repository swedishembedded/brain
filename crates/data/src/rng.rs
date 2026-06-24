// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Deterministic PRNG shared by every dataset generator.
//!
//! We cannot byte-reproduce Python's `random`/`numpy` streams, so brain's
//! datasets are *functionally* equivalent to nanogpt's (same task, format, and
//! distributions) rather than bit-identical. What matters for the benchmark is
//! that a fixed `seed` always yields the same corpus — that holds here.
//!
//! Algorithm: SplitMix64 (Steele et al.) — tiny, fast, and well-distributed.

/// A seedable SplitMix64 generator.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Seed the generator. Any seed (including 0) is valid.
    pub fn new(seed: u64) -> Self {
        Rng {
            // Offset so seed 0 still produces a well-mixed stream.
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    /// Next raw 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f64` in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        // Top 53 bits -> mantissa.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform `f32` in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        self.next_f64() as f32
    }

    /// Uniform integer in `[lo, hi]` **inclusive** (matches Python `randint`).
    pub fn gen_range_inclusive(&mut self, lo: i64, hi: i64) -> i64 {
        debug_assert!(lo <= hi);
        let span = (hi - lo + 1) as u64;
        lo + (self.next_u64() % span) as i64
    }

    /// Uniform `f64` in `[lo, hi)` (matches Python `uniform`).
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// Pick a uniformly random element of a non-empty slice.
    pub fn choice<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.gen_range_inclusive(0, items.len() as i64 - 1) as usize]
    }

    /// Standard-normal `f64` via Box–Muller (used by the time-series generator).
    pub fn next_gaussian(&mut self) -> f64 {
        // Avoid log(0).
        let u1 = (self.next_f64()).max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_for_fixed_seed() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn range_is_inclusive_and_in_bounds() {
        let mut r = Rng::new(7);
        for _ in 0..10_000 {
            let v = r.gen_range_inclusive(3, 9);
            assert!((3..=9).contains(&v));
        }
    }

    #[test]
    fn floats_in_unit_interval() {
        let mut r = Rng::new(1);
        for _ in 0..10_000 {
            let f = r.next_f64();
            assert!((0.0..1.0).contains(&f));
        }
    }
}

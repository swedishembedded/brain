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

/// The deterministic LCG that fixtures, parity probes and kernel tests use to
/// fill buffers without pulling in `rand` — and the sanctioned home for
/// PRODUCTION deterministic init too (e.g. seeding a LoRA/adapter weight):
/// audit F40 found post-unification hand-rolled copies growing back precisely
/// because this doc used to scope the type to tests only, leaving production
/// init with no named home. If a stream must be *statistically stronger* than
/// an LCG, use [`Rng`] (SplitMix64) — never a fresh local copy of either.
///
/// It is a **separate type from [`Rng`] on purpose**: [`Rng`] is SplitMix64 and
/// defines the on-disk datasets, so its stream must never move. This one exists
/// so the ~40 hand-rolled copies of
///
/// ```text
/// *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
/// ((*s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
/// ```
///
/// have one home. That expression is **one-sided**: `u64 >> 33` is 31 bits, so
/// `/ 2^31` lands in `[0, 1)` and the `- 1.0` makes every sample land in
/// `[-1, 0)`. The intended `[-1, 1)` never occurred, which meant no test ever
/// fed a positive value to `relu`/`prelu`/`leaky_relu`/`max`-style kernels —
/// exactly the branch worth testing. [`Lcg::signed`] shifts by **32**, keeping
/// 32 bits, so `/ 2^31 - 1.0` covers the full `[-1, 1)`.
///
/// The LCG constants are Knuth's MMIX; the top bits are used because an LCG's
/// low bits have short periods.
#[derive(Clone, Debug)]
pub struct Lcg {
    state: u64,
}

impl Lcg {
    /// Seed the generator. The first value returned is the seed *advanced once*,
    /// matching the hand-rolled helpers this replaces.
    pub fn new(seed: u64) -> Self {
        Lcg { state: seed }
    }

    /// Next raw 64-bit state.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    /// The top 32 bits of the next state — the raw source for the float helpers.
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Uniform `f32` in `[0, 1)`.
    pub fn unit(&mut self) -> f32 {
        self.next_u32() as f32 / (1u64 << 32) as f32
    }

    /// Uniform `f32` in `[-1, 1)` — **both** signs, unlike the copies this
    /// replaces.
    pub fn signed(&mut self) -> f32 {
        self.next_u32() as f32 / (1u64 << 31) as f32 - 1.0
    }

    /// Uniform `f32` in `[-a, a)`.
    pub fn scaled(&mut self, a: f32) -> f32 {
        self.signed() * a
    }

    /// `n` samples from [`Lcg::signed`].
    pub fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.signed()).collect()
    }

    /// `n` samples from [`Lcg::scaled`].
    pub fn vec_scaled(&mut self, n: usize, a: f32) -> Vec<f32> {
        (0..n).map(|_| self.scaled(a)).collect()
    }

    /// `n` samples from [`Lcg::unit`].
    pub fn vec_unit(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.unit()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The defect this type exists to fix: the stream must straddle zero.
    #[test]
    fn signed_covers_both_halves_of_minus_one_to_one() {
        let mut r = Lcg::new(1);
        let (mut lo, mut hi) = (0usize, 0usize);
        for _ in 0..10_000 {
            let v = r.signed();
            assert!((-1.0..1.0).contains(&v), "out of range: {v}");
            if v < 0.0 {
                lo += 1;
            } else {
                hi += 1;
            }
        }
        // A fair generator gives ~5000/5000; assert only that neither half is
        // starved, which the `>> 33` version failed with hi == 0.
        assert!(lo > 4000 && hi > 4000, "one-sided stream: {lo} negative / {hi} non-negative");
    }

    #[test]
    fn unit_stays_in_zero_to_one() {
        let mut r = Lcg::new(9);
        for _ in 0..10_000 {
            let v = r.unit();
            assert!((0.0..1.0).contains(&v), "out of range: {v}");
        }
    }

    #[test]
    fn lcg_is_deterministic_for_fixed_seed() {
        let (mut a, mut b) = (Lcg::new(3), Lcg::new(3));
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

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

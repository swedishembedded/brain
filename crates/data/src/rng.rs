// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Deterministic PRNG shared by every dataset generator.
//!
//! We cannot byte-reproduce Python's `random`/`numpy` streams, so brain's
//! datasets are *functionally* equivalent to nanogpt's (same task, format, and
//! distributions) rather than bit-identical. What matters for the benchmark is
//! that a fixed `seed` always yields the same corpus — that holds here.
//!
//! Algorithm: SplitMix64 (fast splittable generator) - tiny, fast, and well-distributed.
//!
//! [`Rng::new`] is deterministic: the seed it is given is the whole stream. A
//! caller that has no seed to give does not pass a constant, it draws one with
//! [`random_seed`] and reports it so the run can be reproduced. That holds for
//! a library option struct's `Default` as much as for a CLI flag left unset;
//! a fixed seed belongs only to a caller that needs a fixed stream (a test, a
//! frozen artifact) and says why.

/// Where an unset seed comes from; re-exported so existing callers keep the
/// `data::rng::random_seed` path.
pub use seed::random_seed;

/// A seedable SplitMix64 generator.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Seed the generator. Any seed (including 0) is valid.
    pub fn new(seed: u64) -> Self {
        Rng {
            // One step ahead of the textbook SplitMix64 seed: `new(s)` skips the first
            // output of the plain generator. SplitMix64 mixes every state, including
            // seed 0, so the offset is not needed for quality; it stays because every
            // on-disk dataset was generated from this stream.
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    /// A generator on a freshly drawn seed, for a caller that needs no
    /// particular stream (a test whose assertions hold for any input). The
    /// seed is printed to stderr, which libtest shows for a failing test, so
    /// the failure reproduces with [`Rng::new`] of that value.
    pub fn random() -> Self {
        let seed = random_seed();
        eprintln!("Rng seed: {seed}");
        Rng::new(seed)
    }

    /// The generator's whole state: [`Rng::from_state`] of it continues the
    /// stream exactly where this one is, which is what an interrupted
    /// training run needs to draw the same batches after it resumes.
    pub fn state(&self) -> u64 {
        self.state
    }

    /// The generator [`Rng::state`] was read from, at the same point in its
    /// stream. Not a seed: `Rng::new(s)` and `Rng::from_state(s)` differ.
    pub fn from_state(state: u64) -> Self {
        Rng { state }
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
    /// Raw values that would make some results more likely than others are
    /// redrawn (probability under `span / 2^64`), so the stream is that of
    /// plain `% span` except at those draws.
    pub fn gen_range_inclusive(&mut self, lo: i64, hi: i64) -> i64 {
        debug_assert!(lo <= hi);
        // Wrapping: the span of the whole i64 range is 2^64, which reads as 0 and
        // means "every raw value is in range".
        let span = (hi.wrapping_sub(lo) as u64).wrapping_add(1);
        if span == 0 {
            return lo.wrapping_add(self.next_u64() as i64);
        }
        lo + bounded(span, || self.next_u64()) as i64
    }

    /// Uniform `f64` in `[lo, hi)` (matches Python `uniform`).
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// Pick a uniformly random element of a non-empty slice.
    pub fn choice<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.gen_range_inclusive(0, items.len() as i64 - 1) as usize]
    }

    /// Standard-normal `f64`: the cosine half of Box-Muller, with the sine
    /// partner discarded so each call consumes two uniforms (used by the
    /// time-series generator). Unbounded: a zero first uniform is redrawn.
    pub fn next_gaussian(&mut self) -> f64 {
        gaussian_from(|| self.next_f64())
    }
}

/// The largest raw value accepted when drawing below `span` from a source whose
/// values are uniform on `[0, max]`. The values above it are the partial last
/// block of residues, which `% span` would hit once more often than the rest.
fn accept_limit(span: u64, max: u64) -> u64 {
    // `max + 1` values do not divide evenly by `span`; the remainder is the
    // size of the partial block.
    let partial = (max % span + 1) % span;
    max - partial
}

/// Uniform in `[0, span)` from a source of raw 64-bit values, `span >= 1`.
/// A raw value in the partial last block is redrawn, so every residue is
/// equally likely; any other value gives the same residue plain `%` would.
fn bounded(span: u64, mut next: impl FnMut() -> u64) -> u64 {
    let limit = accept_limit(span, u64::MAX);
    loop {
        let x = next();
        if x <= limit {
            return x % span;
        }
    }
}

/// The cosine half of Box-Muller from a source of uniforms on `[0, 1)`. A zero
/// first uniform has no logarithm and is redrawn rather than capped, so the
/// output is not bounded by a clamp.
fn gaussian_from(mut uniform: impl FnMut() -> f64) -> f64 {
    let u1 = loop {
        let u = uniform();
        if u > 0.0 {
            break u;
        }
    };
    let u2 = uniform();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// The deterministic LCG that fixtures, parity probes and kernel tests use to
/// fill buffers without pulling in `rand` — and the sanctioned home for
/// PRODUCTION deterministic init too (e.g. seeding a LoRA/adapter weight):
/// audit F40 found post-unification ad-hoc copies growing back precisely
/// because this doc used to scope the type to tests only, leaving production
/// init with no named home. If a stream must be *statistically stronger* than
/// an LCG, use [`Rng`] (SplitMix64) — never a fresh local copy of either.
///
/// It is a **separate type from [`Rng`] on purpose**: [`Rng`] is SplitMix64 and
/// defines the on-disk datasets, so its stream must never move. This one exists
/// so the ~40 ad-hoc copies of
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
    /// matching the ad-hoc helpers this replaces.
    pub fn new(seed: u64) -> Self {
        Lcg { state: seed }
    }

    /// A generator on a freshly drawn seed, for a caller that needs no
    /// particular stream (a test whose assertions hold for any input). The
    /// seed is printed to stderr, which libtest shows for a failing test, so
    /// the failure reproduces with [`Lcg::new`] of that value.
    pub fn random() -> Self {
        let seed = random_seed();
        eprintln!("Lcg seed: {seed}");
        Lcg::new(seed)
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

    /// A test that needs no particular stream draws a fresh one per call.
    #[test]
    fn random_generators_do_not_repeat() {
        assert_ne!(Rng::random().next_u64(), Rng::random().next_u64());
        assert_ne!(Lcg::random().next_u64(), Lcg::random().next_u64());
    }

    /// A generator rebuilt from a saved state draws exactly what the
    /// original would have drawn next.
    #[test]
    fn a_saved_state_continues_the_stream() {
        let mut original = Rng::new(random_seed());
        original.next_u64();
        let mut resumed = Rng::from_state(original.state());
        for _ in 0..8 {
            assert_eq!(resumed.next_u64(), original.next_u64());
        }
    }

    /// The defect this type exists to fix: the stream must straddle zero.
    #[test]
    fn signed_covers_both_halves_of_minus_one_to_one() {
        let mut r = Lcg::new(random_seed());
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
        let mut r = Lcg::new(random_seed());
        for _ in 0..10_000 {
            let v = r.unit();
            assert!((0.0..1.0).contains(&v), "out of range: {v}");
        }
    }

    #[test]
    fn lcg_is_deterministic_for_fixed_seed() {
        let seed = random_seed();
        let (mut a, mut b) = (Lcg::new(seed), Lcg::new(seed));
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let seed = random_seed();
        let mut a = Rng::new(seed);
        let mut b = Rng::new(seed);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn range_is_inclusive_and_in_bounds() {
        let mut r = Rng::new(random_seed());
        for _ in 0..10_000 {
            let v = r.gen_range_inclusive(3, 9);
            assert!((3..=9).contains(&v));
        }
    }

    #[test]
    fn floats_in_unit_interval() {
        let mut r = Rng::new(random_seed());
        for _ in 0..10_000 {
            let f = r.next_f64();
            assert!((0.0..1.0).contains(&f));
        }
    }

    /// Every residue of a span that does not divide 2^64 must be equally
    /// likely: for span 3 the raw value `u64::MAX` (residue 0 of the partial
    /// last block) is rejected and the next draw is used.
    #[test]
    fn bounded_rejects_the_partial_block_at_the_top() {
        // 2^64 = 3 * 6148914691236517205 + 1, so only `u64::MAX` is in the partial block.
        let mut draws = vec![u64::MAX, 7].into_iter();
        let v = bounded(3, || draws.next().expect("one redraw"));
        assert_eq!(v, 7 % 3);
        // Every value below the block is accepted on the first draw.
        assert_eq!(bounded(3, || u64::MAX - 1), (u64::MAX - 1) % 3);
    }

    /// Exhaustive check on a scaled-down generator: with a 8-bit source and a span of 100,
    /// the accepted draws hit every residue exactly equally often.
    #[test]
    fn bounded_accepts_each_residue_equally_often() {
        let span = 100u64;
        let mut hits = [0u32; 100];
        for raw in 0..=u8::MAX as u64 {
            let limit = accept_limit(span, u8::MAX as u64);
            if raw <= limit {
                hits[(raw % span) as usize] += 1;
            }
        }
        assert!(hits.iter().all(|&h| h == hits[0]), "{hits:?}");
        assert_eq!(hits[0], 2);
    }

    /// A zero first uniform has no logarithm: it is redrawn, not capped, so
    /// no output is bounded by a clamp.
    #[test]
    fn gaussian_redraws_a_zero_uniform() {
        let mut draws = vec![0.0, 0.0, 0.25, 0.0].into_iter();
        let z = gaussian_from(|| draws.next().expect("three draws"));
        let want = (-2.0 * 0.25f64.ln()).sqrt();
        assert!((z - want).abs() < 1e-12, "{z} vs {want}");
    }

    /// A tiny positive uniform is used as drawn (the old clamp raised it to 1e-12).
    #[test]
    fn gaussian_does_not_clamp_a_small_uniform() {
        let u1 = 1e-15;
        let mut draws = vec![u1, 0.0].into_iter();
        let z = gaussian_from(|| draws.next().unwrap());
        assert!((z - (-2.0 * u1.ln()).sqrt()).abs() < 1e-9);
        assert!(z > 8.0, "tail beyond 7.43 sigma must be reachable: {z}");
    }

    /// The pinned stream of accepted draws is unchanged by the rejection.
    #[test]
    fn range_matches_plain_modulo_when_nothing_is_rejected() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            let want = 5 + (b.next_u64() % 17) as i64;
            assert_eq!(a.gen_range_inclusive(5, 21), want);
        }
    }

    #[test]
    fn full_i64_range_does_not_overflow() {
        let mut r = Rng::new(1);
        let _ = r.gen_range_inclusive(i64::MIN, i64::MAX);
        assert_eq!(r.gen_range_inclusive(7, 7), 7);
    }
}

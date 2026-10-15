// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Fixture for scripts/gates/check-seed-literals.sh: one allowlisted
//! production literal, one test literal with its reason, and strings and
//! comments that must not count.

/// A salt: mixed into a caller's seed, never a seed itself.
pub const STREAM_SEED_SALT: u64 = 0x5eed;

/// `seed: 99` and `Rng::new(5)` in a comment are not literals.
pub fn describe() -> &'static str {
    "seed: 99 inside a string is not a literal either"
}

#[cfg(test)]
mod tests {
    #[test]
    fn replays() {
        // seed: the expected value below was recorded with this stream
        let _ = Rng::new(7);
    }
}

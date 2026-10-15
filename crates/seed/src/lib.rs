// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible training and evaluation
// pipelines for its clients. If your team needs expertise in seeding,
// reproducibility and experiment provenance for machine learning then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Where a run's seed comes from.
//!
//! [`random_seed`] draws the seed of a run whose caller gave none, and
//! [`parse_seed`] reads one a caller did give. This is a leaf crate with no
//! dependencies so that the data crate, the capability interface and the
//! forecast crate (which depends on no other brain crate) can all resolve a
//! missing seed the same way.

/// A real (non-reproducible) `u64` seed for a run whose caller gave none.
///
/// Every default seed in this workspace is random: a CLI flag or a served
/// request's `seed` left unset, and equally a library option struct's
/// `Default`. Two back-to-back unseeded runs must differ, because a hardcoded
/// default makes them byte-identical and so indistinguishable from a caller
/// who deliberately asked for seed `0`. The drawn value is then printed at the
/// boundary and recorded in the run's output, so passing it back reproduces
/// the run. This is the ONE shared source of entropy for that resolution.
///
/// `std`-only (no `rand`/`getrandom` dependency): mixes wall-clock
/// nanoseconds, the process id and a per-process atomic counter (so two calls
/// in the same nanosecond-resolution tick, or two processes started in the
/// same tick, still diverge) through a SplitMix64 finalizer, additionally
/// XORing in 8 bytes from `/dev/urandom` on Unix when available -
/// best-effort, so a chroot/container without it still returns a value
/// instead of panicking.
///
/// NOT reproducible and not itself a PRNG: a caller who needs a fixed stream
/// seeds the generator explicitly, either with a value of their own or with
/// the one this function just produced.
pub fn random_seed() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let pid = std::process::id() as u64;
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut mixed = nanos ^ pid.rotate_left(17) ^ count.wrapping_mul(0x9E37_79B9_7F4A_7C15);

    #[cfg(unix)]
    {
        use std::io::Read;
        if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
            let mut buf = [0u8; 8];
            if f.read_exact(&mut buf).is_ok() {
                mixed ^= u64::from_le_bytes(buf);
            }
        }
    }

    // SplitMix64 finalizer (same as `Rng::next_u64`) so the mixed bits are
    // well-distributed rather than passed straight through.
    let mut z = mixed;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Parse a seed a caller passed: decimal (`42`) or `0x`-prefixed hexadecimal
/// (`0xdeadbeef`), the full `u64` range. A negative number, a value above
/// `u64::MAX`, an empty string and anything else is refused with a message
/// that quotes the text, so the caller can name the flag it came from.
pub fn parse_seed(text: &str) -> Result<u64, String> {
    let trimmed = text.trim();
    let parsed = match trimmed.strip_prefix("0x").or_else(|| trimmed.strip_prefix("0X")) {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => trimmed.parse::<u64>(),
    };
    parsed.map_err(|_| format!("{text:?} is not a seed (a non-negative integer up to {}, decimal or 0x-hex)", u64::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two back-to-back resolutions of an unset seed must not collide. The
    /// per-process counter mixed in makes that a guarantee, not a statistical
    /// hope, so this is not a flaky test.
    #[test]
    fn random_seed_differs_across_consecutive_calls() {
        let a = random_seed();
        let b = random_seed();
        assert_ne!(a, b, "two consecutive unseeded resolutions must not collide");
    }

    #[test]
    fn parse_seed_reads_decimal_and_hex_over_the_full_u64_range() {
        assert_eq!(parse_seed("0"), Ok(0));
        assert_eq!(parse_seed("1337"), Ok(1337));
        assert_eq!(parse_seed("0xff"), Ok(255));
        assert_eq!(parse_seed("18446744073709551615"), Ok(u64::MAX));
        assert_eq!(parse_seed("0xFFFFFFFFFFFFFFFF"), Ok(u64::MAX));
    }

    #[test]
    fn parse_seed_refuses_what_is_not_a_seed() {
        for bad in ["", "abc", "-1", "1.5", "0x", "18446744073709551616", "0x10000000000000000"] {
            let err = parse_seed(bad).expect_err(bad);
            assert!(err.contains(&format!("{bad:?}")), "the message quotes the text: {err}");
        }
    }
}

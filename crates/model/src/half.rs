// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! bf16/f16 host-side pack/unpack - the CPU half of B4's/B5's storage tiers
//! (`kernels::template::dtype_variant`'s device-side decode is the other
//! half; `crates/model/tests/bf16_roundtrip.rs`/`f16_roundtrip.rs` prove the
//! two agree end to end).
//!
//! **Bit layout - must match `dtype_variant`'s decode exactly.** A bf16 value
//! is the top 16 bits of an f32 (`checkpoint::safetensors::bf16_to_f32`:
//! `f32::from_bits((h as u32) << 16)`, the existing read-path convention this
//! module reuses verbatim rather than inventing a second one). [`pack_bf16`]
//! packs two consecutive elements per `u32` word: element `2i` (even) in the
//! LOW 16 bits, element `2i+1` (odd) in the HIGH 16 bits. `dtype_variant`'s
//! WGSL decode for index `IDENT` reads word `IDENT >> 1u` and shifts right by
//! `16u * (IDENT & 1u)` before masking - i.e. even indices (bit 0 = 0) read
//! the low half (shift 0), odd indices read the high half (shift 16). Same
//! convention, checked at the bit level by this module's own tests and
//! end-to-end by the dual-backend roundtrip test.
//!
//! **f16 (B5)** reuses the exact same packed-word convention -
//! [`pack_f16`]/[`f32_to_f16`] pack/round a real IEEE-754 binary16 bit
//! pattern (not bf16's truncated-f32 shortcut - f16's 5-bit exponent needs
//! real re-biasing, done device-side by `dtype_variant`'s magic-multiply
//! decode expression), but the two-per-`u32`, low-half-is-even-index
//! packing is identical to bf16's, for consistency. Unlike [`f32_to_bf16`]
//! (hand-rolled - bf16 narrowing is a one-shift RNE truncation with no
//! crate needed), [`f32_to_f16`] delegates to the `half` crate's
//! `f16::from_f32` - already a workspace dependency (`crates/checkpoint`
//! uses it for the safetensors f16 reader/writer) and a well-tested library
//! implementation is strictly preferable to a second hand-rolled
//! round-to-nearest-even/overflow/subnormal routine for the HOST side of
//! this conversion. The WGSL DEVICE-side decode
//! (`kernels::template::dtype_variant`'s f16 arm, `crates/kernels/src/
//! template.rs`) is still hand-written and verified independently (an
//! exhaustive 65536-pattern host-side check against a non-bit-trick
//! reference) - no crate can help inside a shader.

/// Round `f` to bf16 with round-to-nearest-even (RNE) on the 16 low bits
/// being discarded - NOT truncation. Truncation always rounds toward zero, so
/// every packed weight would be silently biased low in magnitude; RNE is what
/// real bf16 hardware/compilers do and is required for the roundtrip test's
/// tolerance math to hold (see that test's own comment).
///
/// NaN payloads are preserved as NaN (forcing the top mantissa bit on, so a
/// NaN can never truncate into an `Inf`); `+-Inf` and finite values round via
/// the standard "add the rounding bias, then truncate" trick, which is exact
/// because bf16 shares f32's exponent field - only the mantissa narrows.
pub fn f32_to_bf16(f: f32) -> u16 {
    let bits = f.to_bits();
    if f.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    // Round-half-to-even: add 0x7FFF plus the low bit of the bits about to
    // survive (bit 16) as the tie-breaker, then the truncating shift below
    // rounds correctly in every case, including ties.
    let bias = 0x7FFFu32 + ((bits >> 16) & 1);
    let rounded = bits.wrapping_add(bias);
    (rounded >> 16) as u16
}

/// Pack `f32s` two-per-`u32` as bf16 (RNE - see [`f32_to_bf16`]): element
/// `2i` in the low 16 bits, element `2i+1` in the high 16 bits (module doc
/// comment). An odd-length input's final word has its unused high half
/// zeroed - `dtype_variant`'s decode never reads past the caller's own
/// logical element count, so that padding is never observed.
pub fn pack_bf16(f32s: &[f32]) -> Vec<u32> {
    f32s.chunks(2)
        .map(|pair| {
            let lo = f32_to_bf16(pair[0]) as u32;
            let hi = pair.get(1).map(|&v| f32_to_bf16(v) as u32).unwrap_or(0);
            lo | (hi << 16)
        })
        .collect()
}

/// Round `f` to an f16 (binary16) bit pattern - round-to-nearest-even,
/// correct saturation to +-infinity on overflow, correct flush to (signed)
/// subnormal/zero on underflow. Delegates to the `half` crate's
/// `f16::from_f32` rather than hand-rolling a second implementation of this
/// real, fiddly conversion - see this module's doc comment for why that is
/// the right call for the HOST side specifically (the WGSL device-side
/// decode has no such option).
pub fn f32_to_f16(f: f32) -> u16 {
    ::half::f16::from_f32(f).to_bits()
}

/// Pack `f32s` two-per-`u32` as f16 (RNE - see [`f32_to_f16`]): element `2i`
/// in the low 16 bits, element `2i+1` in the high 16 bits - the SAME
/// low/high-half convention [`pack_bf16`] uses, for consistency (both are
/// read by `dtype_variant`'s identical `IDENT >> 1u` / `IDENT & 1u` packed-
/// word selection; only the decode expression differs per tier). An
/// odd-length input's final word has its unused high half zeroed -
/// `dtype_variant`'s decode never reads past the caller's own logical
/// element count, so that padding is never observed.
pub fn pack_f16(f32s: &[f32]) -> Vec<u32> {
    f32s.chunks(2)
        .map(|pair| {
            let lo = f32_to_f16(pair[0]) as u32;
            let hi = pair.get(1).map(|&v| f32_to_f16(v) as u32).unwrap_or(0);
            lo | (hi << 16)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bit patterns pinned against `checkpoint::safetensors`'s existing
    /// `bf16_and_f32_roundtrip` test fixture (1.0 -> 0x3F80, -4.0 -> 0xC080)
    /// so the two crates' conventions are provably the same, not just
    /// independently self-consistent.
    #[test]
    fn f32_to_bf16_matches_the_safetensors_reader_convention() {
        assert_eq!(f32_to_bf16(1.0), 0x3F80);
        assert_eq!(f32_to_bf16(-4.0), 0xC080);
        assert_eq!(f32_to_bf16(0.0), 0x0000);
        assert_eq!(f32_to_bf16(-0.0), 0x8000);
    }

    /// bf16 is exact for any value whose low 16 mantissa bits are already
    /// zero (every f32 that started life as a widened bf16) -- exercises the
    /// same values the round-trip test packs.
    #[test]
    fn f32_to_bf16_is_exact_for_already_bf16_representable_values() {
        for v in [1.5f32, -2.0, 0.25, 100.0, -0.001, 12345.0] {
            let widened = f32::from_bits((v.to_bits() >> 16) << 16);
            assert_eq!(f32_to_bf16(widened), (widened.to_bits() >> 16) as u16);
        }
    }

    /// Round-to-nearest-EVEN, pinned at an exact tie: `0x3F80_8000` sits
    /// exactly halfway between bf16 `0x3F80` and `0x3F81`; RNE must round to
    /// the EVEN mantissa (`0x3F80`), not always up (truncation would instead
    /// give `0x3F80` here too by luck -- the next test below is the one that
    /// actually distinguishes RNE from truncation).
    #[test]
    fn f32_to_bf16_rounds_ties_to_even() {
        let tie_down = f32::from_bits(0x3F80_8000); // -> 0x3F80 (even) not 0x3F81
        assert_eq!(f32_to_bf16(tie_down), 0x3F80);
        let tie_up = f32::from_bits(0x3F81_8000); // -> 0x3F82 (even) not 0x3F81
        assert_eq!(f32_to_bf16(tie_up), 0x3F82);
    }

    /// The case that actually distinguishes RNE from plain truncation: a
    /// value just ABOVE the halfway point must round UP, which truncation
    /// (which always discards the low bits) would get wrong.
    #[test]
    fn f32_to_bf16_rounds_up_past_the_halfway_point() {
        let just_over_half = f32::from_bits(0x3F80_8001);
        assert_eq!(f32_to_bf16(just_over_half), 0x3F81);
    }

    /// NaN stays NaN through the bf16 narrowing (never silently becomes Inf).
    #[test]
    fn f32_to_bf16_preserves_nan() {
        let bf = f32_to_bf16(f32::NAN);
        let widened = f32::from_bits((bf as u32) << 16);
        assert!(widened.is_nan());
    }

    /// [`pack_bf16`]'s low/high half convention, decoded by hand via the
    /// SAME `(h as u32) << 16` expression `checkpoint::safetensors::
    /// bf16_to_f32` uses (that function is crate-private, so this reproduces
    /// it inline rather than depending on it -- the point is the bit
    /// arithmetic agrees, not the call site).
    #[test]
    fn pack_bf16_low_half_is_even_index_high_half_is_odd_index() {
        let packed = pack_bf16(&[1.0, -4.0, 2.0]);
        assert_eq!(packed.len(), 2, "3 elements -> 2 words (last word's high half padded)");
        let decode = |h: u16| f32::from_bits((h as u32) << 16);
        // word 0: low = elem 0 (1.0), high = elem 1 (-4.0).
        assert_eq!(decode((packed[0] & 0xFFFF) as u16), 1.0);
        assert_eq!(decode((packed[0] >> 16) as u16), -4.0);
        // word 1: low = elem 2 (2.0), high = padding (must decode to +0.0,
        // never read by a correctly-bounded caller, but must not be garbage).
        assert_eq!(decode((packed[1] & 0xFFFF) as u16), 2.0);
        assert_eq!(decode((packed[1] >> 16) as u16), 0.0);
    }

    #[test]
    fn pack_bf16_round_trips_within_bf16_precision() {
        let input: Vec<f32> = (0..37).map(|i| (i as f32) * 0.37 - 5.0).collect();
        let packed = pack_bf16(&input);
        let decode = |h: u16| f32::from_bits((h as u32) << 16);
        for (i, &orig) in input.iter().enumerate() {
            let word = packed[i / 2];
            let half = if i % 2 == 0 { word & 0xFFFF } else { word >> 16 };
            let got = decode(half as u16);
            // bf16 has 7 explicit mantissa bits -> worst-case relative error
            // 2^-8 (one rounding step, half an ULP at the boundary).
            let tol = orig.abs() * 2f32.powi(-8) + 1e-6;
            assert!((got - orig).abs() <= tol, "elem {i}: {orig} -> {got} (tol {tol})");
        }
    }

    // --- B5: f16 (real binary16, not bf16's truncated-f32 shortcut) -------

    /// Decode an f16 bit pattern the same way `half::f16::from_bits(..)
    /// .to_f32()` does - the authoritative reference this module's
    /// [`f32_to_f16`] delegates its encode side to, so a round-trip through
    /// it is checking "did we call the library correctly", not "is the
    /// library's own math right" (that is `kernels::template::
    /// f16_decode_matches_an_independent_reference_for_every_possible_bit_
    /// pattern`'s job, exhaustively, against a from-scratch reference, not
    /// this one).
    fn decode_f16(bits: u16) -> f32 {
        ::half::f16::from_bits(bits).to_f32()
    }

    /// The phase brief's own edge-case table, checked end to end
    /// (`f32_to_f16` then decoded back): zero/negative-zero sign
    /// preservation, +-1.0, the smallest/largest normal, the smallest
    /// subnormal plus a few more, overflow saturating to +-inf (never
    /// wrapping or producing garbage), underflow flushing to signed zero,
    /// +-infinity passing through, and NaN staying SOME NaN (never inf or a
    /// finite number - exact payload not required).
    #[test]
    fn f32_to_f16_round_trips_the_phase_briefs_edge_case_table() {
        // Zero / negative zero: exact, sign preserved.
        assert_eq!(decode_f16(f32_to_f16(0.0)), 0.0);
        assert!(decode_f16(f32_to_f16(0.0)).is_sign_positive());
        assert_eq!(decode_f16(f32_to_f16(-0.0)), 0.0);
        assert!(decode_f16(f32_to_f16(-0.0)).is_sign_negative());

        // +-1.0: exact.
        assert_eq!(decode_f16(f32_to_f16(1.0)), 1.0);
        assert_eq!(decode_f16(f32_to_f16(-1.0)), -1.0);

        // Smallest/largest normal: both exactly representable in f16.
        let smallest_normal = 2f32.powi(-14);
        assert_eq!(decode_f16(f32_to_f16(smallest_normal)), smallest_normal);
        let largest_normal = 65504.0f32;
        assert_eq!(decode_f16(f32_to_f16(largest_normal)), largest_normal);

        // Smallest subnormal and a few more: every f16 subnormal is an
        // integer multiple of 2^-24, exactly representable in f32.
        let smallest_subnormal = 2f32.powi(-24);
        assert_eq!(decode_f16(f32_to_f16(smallest_subnormal)), smallest_subnormal);
        for mult in [2.0, 3.0, 100.0, 1023.0] {
            let v = smallest_subnormal * mult;
            assert_eq!(decode_f16(f32_to_f16(v)), v, "subnormal {mult}x the smallest");
        }

        // Overflow: beyond f16's range must saturate to +-inf, never wrap.
        assert_eq!(decode_f16(f32_to_f16(70000.0)), f32::INFINITY);
        assert_eq!(decode_f16(f32_to_f16(-70000.0)), f32::NEG_INFINITY);

        // Underflow: below the smallest subnormal flushes to signed zero.
        let underflow = 2f32.powi(-25);
        assert_eq!(decode_f16(f32_to_f16(underflow)), 0.0);
        assert!(decode_f16(f32_to_f16(underflow)).is_sign_positive());
        assert_eq!(decode_f16(f32_to_f16(-underflow)), 0.0);
        assert!(decode_f16(f32_to_f16(-underflow)).is_sign_negative());

        // Infinities pass through exactly.
        assert_eq!(decode_f16(f32_to_f16(f32::INFINITY)), f32::INFINITY);
        assert_eq!(decode_f16(f32_to_f16(f32::NEG_INFINITY)), f32::NEG_INFINITY);

        // NaN decodes back to SOME NaN, never inf or a finite number.
        assert!(decode_f16(f32_to_f16(f32::NAN)).is_nan());
    }

    /// Round-to-nearest-EVEN, pinned at an exact tie (the f16 analogue of
    /// [`f32_to_bf16_rounds_ties_to_even`]): f16 truncates 13 low mantissa
    /// bits (23 - 10), so `0x1000` (`2^12`, half of `2^13`) added to an
    /// exactly-representable f16 value's f32 bits sits exactly halfway
    /// between two adjacent f16 mantissas.
    #[test]
    fn f32_to_f16_rounds_ties_to_even() {
        // 1.0 + 2^-11: halfway between f16 mantissa=0 (1.0, even) and
        // mantissa=1 (odd) -- RNE must pick the even one, staying at 1.0.
        let tie_down = f32::from_bits(0x3F80_1000);
        assert_eq!(f32_to_f16(tie_down), 0x3C00);
        // 1.0 + 3*2^-11: halfway between f16 mantissa=1 (odd) and
        // mantissa=2 (even) -- RNE must pick the even one, mantissa=2.
        let tie_up = f32::from_bits(0x3F80_3000);
        assert_eq!(f32_to_f16(tie_up), 0x3C02);
    }

    /// [`pack_f16`]'s low/high half convention - the SAME convention
    /// [`pack_bf16_low_half_is_even_index_high_half_is_odd_index`] pins for
    /// bf16, checked here for f16.
    #[test]
    fn pack_f16_low_half_is_even_index_high_half_is_odd_index() {
        let packed = pack_f16(&[1.0, -4.0, 2.0]);
        assert_eq!(packed.len(), 2, "3 elements -> 2 words (last word's high half padded)");
        assert_eq!(decode_f16((packed[0] & 0xFFFF) as u16), 1.0);
        assert_eq!(decode_f16((packed[0] >> 16) as u16), -4.0);
        assert_eq!(decode_f16((packed[1] & 0xFFFF) as u16), 2.0);
        assert_eq!(decode_f16((packed[1] >> 16) as u16), 0.0);
    }

    #[test]
    fn pack_f16_round_trips_within_f16_precision() {
        let input: Vec<f32> = (0..37).map(|i| (i as f32) * 0.37 - 5.0).collect();
        let packed = pack_f16(&input);
        for (i, &orig) in input.iter().enumerate() {
            let word = packed[i / 2];
            let half = if i % 2 == 0 { word & 0xFFFF } else { word >> 16 };
            let got = decode_f16(half as u16);
            // f16 has 10 explicit mantissa bits -> worst-case relative error
            // 2^-11 (one rounding step, half an ULP at the boundary).
            let tol = orig.abs() * 2f32.powi(-11) + 1e-6;
            assert!((got - orig).abs() <= tol, "elem {i}: {orig} -> {got} (tol {tol})");
        }
    }
}

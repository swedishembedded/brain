// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared int4 (q4) WEIGHT quantization — the host half of the engine's q4
//! inference tier, added for Qwen3.5-35B-A3B (35B params does not fit two
//! 24 GB GPUs even at int8; q4 halves the weight footprint again).
//!
//! **W4A8, not W4A4**: activations stay on the EXISTING int8 dynamic-quant
//! path this crate already ships ([`crate::int8::quant_rows_steps`], backed by
//! `max_abs_row`/`quant_pack`) — only weights are quantized to 4 bits. This
//! keeps every stage but the weight side and the GEMM's inner unpack loop
//! byte-for-byte what int8 already built and validated, which is the whole
//! point of "one implementation for every model": a model that already
//! dispatches the int8 activation-quant kernels adopts q4 weights by relinking
//! one GEMM, not by building a second activation-quant tier. W4A4 was
//! considered and rejected for exactly that reason — it would require a new
//! device-side activation quantizer parallel to the one every int8 model
//! already calls, for a precision tier no int4 accelerator instruction in this
//! engine (there is none — q4 unpacks by hand, see `matmul_q4_dyn.wgsl`)
//! rewards.
//!
//! **Packing**: symmetric GROUP-wise int4 over the same
//! [`crate::int8::GROUP`] = 32-element blocks of the reduction axis
//! [`crate::int8::quantize_weight`] uses (`Q4_0`'s block size is `Q8_0`'s), but
//! 8 values per `u32` instead of 4 - so a group is FOUR words here, and `k`
//! must still be a multiple of 32.
//! `scale s[r, g] = max|w[r, 32g..32g+32]|.max(1e-8) / 7.0` - **7, not 15**:
//! the symmetric 4-bit range is `[-7, 7]`, deliberately not the full
//! two's-complement `[-8, 7]`, for exactly the reason int8 clamps to
//! `[-127, 127]` rather than `[-128, 127]` - a representation symmetric around
//! zero. Nibble `b` of packed word `g` (source columns `[8g, 8g+8)`) occupies
//! bits `[4b, 4b+4)` of `packed[r, g]`, two's-complement (`q as u8 & 0xF`).
//!
//! int4 is NOT exempt from the group-wise rule int8's module doc states - at
//! 15 levels instead of 255, a whole-channel scale spent on one outlier costs
//! MORE, not less. Same group size, same pattern, one fix.
//!
//! The packed layout here is exactly what `matmul_q4*.wgsl` /
//! `moe_linear_gated_q4.wgsl` consume — if it changes, it changes for every
//! model at once.

use crate::int8::GROUP;

/// Packed `u32` words per scale group at 8 nibbles/word (`GROUP / 8` = 4) -
/// int8's [`crate::int8::WORDS_PER_GROUP`] halved, because twice as many
/// values fit a word.
pub const WORDS_PER_GROUP_Q4: usize = GROUP / 8;

/// Group-wise symmetric int4 quantization of an `[n, k]` weight (one scale per
/// [`GROUP`]-element block of the reduction axis), packed into `[n, k/8]` u32
/// (8 int4 nibbles per u32, little-endian along K - nibble `b` in bits
/// `[4b, 4b+4)`). Returns `(packed, scales)` with `scales` shaped
/// `[n, k/GROUP]` row-major and
/// `scales[r, g] = max|w[r, 32g..32g+32]|.max(1e-8) / 7.0`.
/// `k` must be a multiple of [`GROUP`] - the same rule and the same constant
/// as int8, which also subsumes the `% 8` the nibble packing needs.
///
/// Row-parallel for the same reason, and with the same bit-identical
/// guarantee, as [`crate::int8::quantize_weight`] - see that function's doc
/// and the shared gate `tests/quantize_weight_is_schedule_invariant.rs`.
pub fn quantize_weight_q4(w: &[f32], n: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
    assert_eq!(k % GROUP, 0, "q4 K must be a multiple of {GROUP} (got {k})");
    assert_eq!(w.len(), n * k, "weight len {} != n*k {}", w.len(), n * k);
    let kg = k / 8;
    let gs = k / GROUP;
    let mut packed = vec![0u32; n * kg];
    let mut sw = vec![0f32; n * gs];
    backend_cpu::par::chunks2_mut(&mut packed, kg, &mut sw, gs, |r, prow, srow| {
        let row = &w[r * k..r * k + k];
        for (g, s) in srow.iter_mut().enumerate() {
            *s = row[g * GROUP..g * GROUP + GROUP].iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-8) / 7.0;
        }
        for (g, word_out) in prow.iter_mut().enumerate() {
            let inv = 1.0 / srow[g / WORDS_PER_GROUP_Q4];
            let mut word = 0u32;
            for b in 0..8 {
                let q = (row[g * 8 + b] * inv).round().clamp(-7.0, 7.0) as i32;
                word |= ((q as u8) as u32 & 0xF) << (4 * b);
            }
            *word_out = word;
        }
    });
    (packed, sw)
}

/// The exact host-side inverse of [`quantize_weight_q4`]: unpack `[n, k/8]`
/// u32 words back to `[n, k]` f32 via the `[n, k/GROUP]` group scales. Each
/// 4-bit field must be SIGN-EXTENDED (a nibble with bit 3 set is negative) -
/// the single easiest thing to get wrong at 4 bits, see
/// `dequantize_weight_q4_sign_handling_is_load_bearing` below. `k` must be a
/// multiple of [`GROUP`] (mirrors [`quantize_weight_q4`]'s own requirement).
pub fn dequantize_weight_q4(packed: &[u32], sw: &[f32], n: usize, k: usize) -> Vec<f32> {
    assert_eq!(k % GROUP, 0, "q4 K must be a multiple of {GROUP} (got {k})");
    let kg = k / 8;
    let gs = k / GROUP;
    assert_eq!(packed.len(), n * kg, "packed len {} != n*(k/8) {}", packed.len(), n * kg);
    assert_eq!(sw.len(), n * gs, "scale len {} != n*(k/{GROUP}) {}", sw.len(), n * gs);
    let mut w = vec![0f32; n * k];
    for r in 0..n {
        for g in 0..kg {
            let s = sw[r * gs + g / WORDS_PER_GROUP_Q4];
            let word = packed[r * kg + g];
            for b in 0..8 {
                let raw = (word >> (4 * b)) & 0xF;
                // Sign-extend a 4-bit field held in the low bits of a u8:
                // shift it up so its sign bit lands in bit 7, cast to i8 (now
                // sign-correct in the top nibble), then arithmetic-shift back
                // down by 4 — verified against a hand oracle in
                // `sign_extend_nibble_trick_is_correct` below, not just
                // asserted inline.
                let signed = ((raw as u8) << 4) as i8 >> 4;
                w[r * k + g * 8 + b] = signed as f32 * s;
            }
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bit trick [`dequantize_weight_q4`] relies on
    /// (`((raw as u8) << 4) as i8 >> 4`) sign-extends a 4-bit two's-complement
    /// field held in the LOW bits of a byte, checked against a hand oracle for
    /// every one of the 16 possible nibble values — verify the trick, don't
    /// just trust it.
    #[test]
    fn sign_extend_nibble_trick_is_correct() {
        for raw in 0u8..16 {
            let got = ((raw << 4) as i8) >> 4;
            // Hand oracle: values 0..=7 are themselves, 8..=15 are `raw - 16`.
            let want = if raw < 8 { raw as i32 } else { raw as i32 - 16 };
            assert_eq!(got as i32, want, "nibble {raw:#06b}");
        }
    }

    /// Round-trip within one quantization step, per element. 4-bit
    /// quantization is much coarser than int8's: the realistic tolerance
    /// observed at this shape is `sw[r] * 0.5 + 1e-6` (same half-step bound as
    /// int8's own test, just against a 15-level instead of 255-level grid) --
    /// documented explicitly so a future reader does not mistake "4-bit is
    /// lossy" for "the kernel is broken".
    #[test]
    fn round_trips_within_one_step() {
        let (n, k) = (3, 64);
        let w: Vec<f32> = (0..n * k).map(|i| (i as f32 - 20.0) * 0.37).collect();
        let (packed, sw) = quantize_weight_q4(&w, n, k);
        assert_eq!(packed.len(), n * k / 8);
        assert_eq!(sw.len(), n * (k / GROUP));
        for r in 0..n {
            for c in 0..k {
                let word = packed[r * (k / 8) + c / 8];
                let raw = (word >> (4 * (c % 8))) & 0xF;
                let q = ((raw as u8) << 4) as i8 >> 4;
                let s = sw[r * (k / GROUP) + c / GROUP];
                let deq = q as f32 * s;
                assert!((deq - w[r * k + c]).abs() <= s * 0.5 + 1e-6, "r{r} c{c}");
            }
        }
    }

    /// [`dequantize_weight_q4`] must agree with the round-trip math
    /// [`round_trips_within_one_step`] checks inline, over multiple rows AND
    /// multiple scale groups per row, so a slip between the `[n]` shape this
    /// used to have and the `[n, k/32]` it has now cannot pass.
    #[test]
    fn dequantize_weight_q4_matches_quantize_weight_q4_within_one_step() {
        let (n, k) = (5, 96);
        let w: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.11 - 6.0) * (1 + i % 3) as f32).collect();
        let (packed, sw) = quantize_weight_q4(&w, n, k);
        let deq = dequantize_weight_q4(&packed, &sw, n, k);
        assert_eq!(deq.len(), n * k);
        for r in 0..n {
            for c in 0..k {
                let d = deq[r * k + c];
                let orig = w[r * k + c];
                let s = sw[r * (k / GROUP) + c / GROUP];
                assert!((d - orig).abs() <= s * 0.5 + 1e-6, "r{r} c{c}: deq={d} orig={orig} scale={s}");
            }
        }
    }

    /// Mutation-verify: an intentionally wrong "read the nibble as unsigned,
    /// no sign extension" decode must diverge from the correct signed decode
    /// for at least one negative-quantized element. 4-bit sign-extension is
    /// the single easiest thing to get subtly wrong in this file, so a test
    /// that cannot fail this way is not exercising the sign handling at all.
    #[test]
    fn dequantize_weight_q4_sign_handling_is_load_bearing() {
        let (n, k) = (2, 32);
        let w: Vec<f32> = (0..n * k).map(|i| if i % 2 == 0 { -(i as f32) - 1.0 } else { i as f32 + 1.0 }).collect();
        let (packed, sw) = quantize_weight_q4(&w, n, k);
        let correct = dequantize_weight_q4(&packed, &sw, n, k);
        // Deliberately wrong: treat each nibble as UNSIGNED (0..15), dropping
        // the sign-extension `dequantize_weight_q4` relies on.
        let mut wrong = vec![0f32; n * k];
        for r in 0..n {
            for g in 0..k / 8 {
                let s = sw[r * (k / GROUP) + g / WORDS_PER_GROUP_Q4];
                let word = packed[r * (k / 8) + g];
                for b in 0..8 {
                    let q_unsigned = (word >> (4 * b)) & 0xF;
                    wrong[r * k + g * 8 + b] = q_unsigned as f32 * s;
                }
            }
        }
        let mut any_diverged = false;
        for i in 0..w.len() {
            if (correct[i] - wrong[i]).abs() > 1e-3 {
                any_diverged = true;
            }
        }
        assert!(any_diverged, "unsigned-vs-signed nibble reinterpretation should diverge for at least one negative-quantized element");
    }

    /// The W4A8 stride mismatch is the second easiest thing to get wrong
    /// here: x (int8-packed activations) has `k/4` u32 words per row, w
    /// (int4-packed weights) has `k/8` - HALF as many. `k=32` makes a stride
    /// mistake obvious: x has 8 u32s/row, w has 4. This test only checks the
    /// WEIGHT side's own packing density (the GEMM kernels own the mixed-
    /// stride unpack loop, covered by `crates/model/tests/matmul_q4_*.rs`),
    /// but pins the exact word count a caller must budget for `w` - and that
    /// the SCALE count is the same for both tiers, since both group by 32.
    #[test]
    fn q4_word_count_is_half_int8s_at_one_group() {
        let (n, k) = (3, 32);
        let w: Vec<f32> = (0..n * k).map(|i| (i as f32 - 24.0) * 0.5).collect();
        let (packed, sw) = quantize_weight_q4(&w, n, k);
        assert_eq!(packed.len(), n * (k / 8), "q4 packs k/8 = 4 words/row at k=32");
        assert_eq!(packed.len(), n * 4);
        // int8's own packing (crate::int8::quantize_weight) would use k/4 = 8
        // words/row for the same K -- twice the q4 count, spelled out so the
        // ratio is not just asserted as a magic "2".
        let (i8_packed, i8_sw) = crate::int8::quantize_weight(&w, n, k);
        assert_eq!(i8_packed.len(), 2 * packed.len(), "int8 uses 2x q4's word count at equal K (4 vs 8 values/word)");
        assert_eq!(sw.len(), i8_sw.len(), "both tiers group by {GROUP}, so both carry n*(k/32) scales");
    }
}

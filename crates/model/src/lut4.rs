// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Codebook (non-uniform) 4-bit weight quantization - `DType::NF4`/
//! `DType::F4E2M1`'s host half (M8.5).
//!
//! **Identical physical layout to [`crate::int4`]'s `Q4`**: symmetric,
//! GROUP-wise (`crate::int8::GROUP` = 32 elements), 8 codes packed per `u32`
//! nibble, one `f32` scale per group, W4A8 (activations stay on the existing
//! int8 dynamic-quant path - see `crate::int4`'s own module doc for why: no
//! int4 accelerator instruction exists in this engine, so narrowing the
//! activation side too would only add a second quant tier for no reduction
//! in bytes moved). The ONLY difference from `Q4` is the codebook: `Q4`
//! reconstructs `scale * code` (`code` a signed integer in `[-7, 7]`, evenly
//! spaced); this module reconstructs `scale * LUT[code]` (`code` an
//! UNSIGNED index in `[0, 15]`, `LUT` one of the two fixed 16-entry tables
//! below).
//!
//! **Why a non-uniform codebook is worth a second tier at the identical byte
//! budget.** Real weight distributions cluster near zero (roughly bell-shaped,
//! not uniform) - `Q4`'s evenly-spaced `[-7, 7]` grid spends as many codes on
//! the sparsely-populated tails as on the densely-populated middle, so most
//! of its 4 bits are wasted resolving values that occur rarely. `NF4_LUT`'s
//! quantile spacing (dense near zero, sparse at the extremes) puts each code
//! where the probability mass actually is. [`tests::nf4_beats_q4_on_a_
//! gaussian_like_distribution`] measures this directly - not assumed - against
//! a synthetic weight draw shaped like a real checkpoint's own statistics.
//!
//! `F4E2M1_LUT` is a DIFFERENT fixed codebook (the OCP Microscaling FP4
//! format some newer checkpoints ship: 1 sign bit, 2 exponent bits, 1
//! mantissa bit), included for checkpoint compatibility, not because it beats
//! `NF4` on reconstruction error - it is a floating-point grid, not a
//! quantile fit, so it is not expected to.
//!
//! The device kernels (`matmul_q4_gemv_nf4.wgsl`/`matmul_q4_gemv_f4e2m1.wgsl`)
//! reuse `Q4`'s exact nibble-unpack shape, reading each code UNSIGNED (no
//! sign-extension - the codebook itself already carries the sign) and looking
//! it up in a hand-inlined `select` tree matching this file's `NF4_LUT`/
//! `F4E2M1_LUT` order exactly - see those kernels' own headers.

use crate::int8::GROUP;

/// The standard bitsandbytes NF4 codebook (`create_normal_map`'s quantile
/// values), ascending - code `i` (`0..16`) maps to `NF4_LUT[i]`. A well-known
/// public constant (reproduced identically across every open NF4
/// implementation this format has); not derived here, just pinned.
pub const NF4_LUT: [f32; 16] = [
    -1.0,
    -0.696_192_8,
    -0.525_073_05,
    -0.394_917_5,
    -0.284_441_38,
    -0.184_773_43,
    -0.091_050_036,
    0.0,
    0.079_580_3,
    0.160_930_2,
    0.246_112_3,
    0.337_915_24,
    0.440_709_83,
    0.562_617,
    0.722_956_84,
    1.0,
];

/// The OCP Microscaling FP4 (E2M1) codebook: 1 sign bit (nibble bit 3) + 2
/// exponent bits + 1 mantissa bit (bias 1). Magnitudes `{0, 0.5, 1, 1.5, 2,
/// 3, 4, 6}` for the low 3 bits (`(1 + mantissa*0.5) * 2^(exp-1)` for
/// `exp>=1`, `mantissa*0.5` for `exp==0`), sign applied by bit 3 - so
/// `F4E2M1_LUT[code]` is exactly `sign(code) * magnitude(code & 0x7)`, spelled
/// out as a flat table because the device kernel needs the SAME table, not
/// the formula.
pub const F4E2M1_LUT: [f32; 16] =
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

/// Group-wise codebook quantization of an `[n, k]` weight, following
/// [`crate::int4::quantize_weight_q4`]'s packing exactly (`k` a multiple of
/// [`GROUP`], 8 codes/`u32`, one scale per [`GROUP`]-element block) but
/// choosing each element's code by NEAREST-NEIGHBOUR search in `lut`
/// (unsigned index, `0..16`) instead of `Q4`'s uniform `round`.
/// `scale[r, g] = max|w[r, 32g..32g+32]|.max(1e-8) / lut_absmax`, so the
/// group's largest-magnitude element lands exactly on `lut`'s own extreme
/// entry (matching `Q4`'s own "outlier sets the scale" convention).
///
/// Row-parallel, same shape and the same bit-identical-across-schedules
/// guarantee as [`crate::int4::quantize_weight_q4`] (each row's own scale and
/// packed words depend only on that row's own input slice).
pub fn quantize_weight_lut4(w: &[f32], n: usize, k: usize, lut: &[f32; 16]) -> (Vec<u32>, Vec<f32>) {
    assert_eq!(k % GROUP, 0, "lut4 K must be a multiple of {GROUP} (got {k})");
    assert_eq!(w.len(), n * k, "weight len {} != n*k {}", w.len(), n * k);
    let lut_absmax = lut.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-8);
    let kg = k / 8;
    let gs = k / GROUP;
    let mut packed = vec![0u32; n * kg];
    let mut sw = vec![0f32; n * gs];
    backend_cpu::par::chunks2_mut(&mut packed, kg, &mut sw, gs, |r, prow, srow| {
        let row = &w[r * k..r * k + k];
        for (g, s) in srow.iter_mut().enumerate() {
            let amax = row[g * GROUP..g * GROUP + GROUP].iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-8);
            *s = amax / lut_absmax;
        }
        for (g, word_out) in prow.iter_mut().enumerate() {
            let inv = 1.0 / srow[g / (GROUP / 8)];
            let mut word = 0u32;
            for b in 0..8 {
                let target = row[g * 8 + b] * inv;
                let mut best_code = 0u32;
                let mut best_err = f32::INFINITY;
                for (code, &v) in lut.iter().enumerate() {
                    let err = (target - v).abs();
                    if err < best_err {
                        best_err = err;
                        best_code = code as u32;
                    }
                }
                word |= best_code << (4 * b);
            }
            *word_out = word;
        }
    });
    (packed, sw)
}

/// The exact host-side inverse of [`quantize_weight_lut4`]: unpack `[n, k/8]`
/// `u32` words back to `[n, k]` `f32` via the `[n, k/GROUP]` group scales and
/// `lut`. Each 4-bit field is read UNSIGNED (`0..16`, no sign-extension -
/// unlike [`crate::int4::dequantize_weight_q4`], the codebook itself carries
/// the sign, the nibble is a plain array index).
pub fn dequantize_weight_lut4(packed: &[u32], sw: &[f32], n: usize, k: usize, lut: &[f32; 16]) -> Vec<f32> {
    assert_eq!(k % GROUP, 0, "lut4 K must be a multiple of {GROUP} (got {k})");
    let kg = k / 8;
    let gs = k / GROUP;
    assert_eq!(packed.len(), n * kg, "packed len {} != n*(k/8) {}", packed.len(), n * kg);
    assert_eq!(sw.len(), n * gs, "scale len {} != n*(k/{GROUP}) {}", sw.len(), n * gs);
    let mut w = vec![0f32; n * k];
    for r in 0..n {
        for g in 0..kg {
            let s = sw[r * gs + g / (GROUP / 8)];
            let word = packed[r * kg + g];
            for b in 0..8 {
                let code = ((word >> (4 * b)) & 0xF) as usize;
                w[r * k + g * 8 + b] = lut[code] * s;
            }
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::int4::quantize_weight_q4;

    #[test]
    fn nf4_lut_is_sorted_ascending_and_spans_minus_one_to_one() {
        for w in NF4_LUT.windows(2) {
            assert!(w[0] < w[1], "NF4_LUT must be strictly ascending: {w:?}");
        }
        assert_eq!(NF4_LUT[0], -1.0);
        assert_eq!(NF4_LUT[15], 1.0);
    }

    #[test]
    fn f4e2m1_lut_matches_the_ocp_e2m1_magnitude_table() {
        // Positive half (code 0..8): magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
        assert_eq!(&F4E2M1_LUT[0..8], &[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]);
        // Negative half (code 8..16): the same magnitudes, negated, same order.
        for i in 0..8 {
            assert_eq!(F4E2M1_LUT[8 + i], -F4E2M1_LUT[i], "code {}", 8 + i);
        }
    }

    #[test]
    fn round_trips_within_the_nearest_codebook_entrys_gap() {
        let (n, k) = (3, 64);
        let w: Vec<f32> = (0..n * k).map(|i| (i as f32 - 20.0) * 0.037).collect();
        let (packed, sw) = quantize_weight_lut4(&w, n, k, &NF4_LUT);
        let deq = dequantize_weight_lut4(&packed, &sw, n, k, &NF4_LUT);
        assert_eq!(deq.len(), n * k);
        // NF4's widest gap between adjacent quantiles bounds the worst-case
        // per-element error at half that gap (times the group's own scale).
        let max_gap = NF4_LUT.windows(2).map(|w| w[1] - w[0]).fold(0f32, f32::max);
        for r in 0..n {
            for c in 0..k {
                let s = sw[r * (k / GROUP) + c / GROUP];
                assert!((deq[r * k + c] - w[r * k + c]).abs() <= s * max_gap * 0.5 + 1e-6, "r{r} c{c}");
            }
        }
    }

    #[test]
    fn dequantize_matches_quantize_for_f4e2m1_too() {
        let (n, k) = (2, 32);
        let w: Vec<f32> = (0..n * k).map(|i| ((i as f32) - 16.0) * 0.4).collect();
        let (packed, sw) = quantize_weight_lut4(&w, n, k, &F4E2M1_LUT);
        let deq = dequantize_weight_lut4(&packed, &sw, n, k, &F4E2M1_LUT);
        let max_gap = F4E2M1_LUT.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0f32, f32::max);
        for i in 0..n * k {
            let s = sw[i / k * (k / GROUP) + (i % k) / GROUP];
            assert!((deq[i] - w[i]).abs() <= s * max_gap * 0.5 + 1e-6, "i{i}");
        }
    }

    #[test]
    fn every_code_is_reachable_not_just_a_subset() {
        // A weight row deliberately constructed to hit every quantile
        // (values placed AT each LUT entry's own scaled position) must
        // decode back to a packed buffer using all 16 codes at least once -
        // catches an off-by-one/truncated search range in the nearest-code
        // loop that a narrower input distribution would miss.
        let (n, k) = (1, 32);
        let mut w = vec![0f32; k];
        for (i, v) in NF4_LUT.iter().enumerate() {
            w[i] = *v; // scale will land at 1.0 since max|w| == 1.0 == lut_absmax
        }
        for v in w.iter_mut().skip(16) {
            *v = 0.0;
        }
        let (packed, _sw) = quantize_weight_lut4(&w, n, k, &NF4_LUT);
        let mut seen = [false; 16];
        for word in &packed {
            for b in 0..8 {
                seen[((word >> (4 * b)) & 0xF) as usize] = true;
            }
        }
        for (code, hit) in seen.iter().enumerate() {
            assert!(*hit, "code {code} never produced by the nearest-neighbour search");
        }
    }

    /// **The actual value proposition, measured, not assumed.** Draws a
    /// synthetic weight distribution shaped like a real checkpoint's own
    /// statistics (roughly Gaussian, small-magnitude, the well-documented
    /// shape trained weights take - see this crate's other quantizers'
    /// tests for the same convention), quantizes it BOTH ways at the
    /// identical 4-bit/32-element budget, and asserts NF4's non-uniform
    /// codebook achieves LOWER mean-squared reconstruction error than Q4's
    /// uniform grid - the whole point of M8.5, checked directly rather than
    /// taken on faith from the codebook's shape.
    #[test]
    fn nf4_beats_q4_on_a_gaussian_like_distribution() {
        let (n, k) = (64, 256);
        // A cheap deterministic approximately-Gaussian generator (sum of
        // uniforms, Irwin-Hall / central-limit-ish), scaled to a realistic
        // trained-weight magnitude (small, std ~0.02) - no external RNG
        // crate needed for a reproducibility-pinned unit test.
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // The top 32 bits (not the low ones, which an LCG's own recurrence
            // correlates most strongly) as a uniform `u32`, scaled to `[0,1)`.
            // An earlier draft shifted by 33 (only 31 significant bits) and
            // divided by `u32::MAX` (32 bits) - silently halving the real
            // range to `[0, 0.5)`, which biased this "Gaussian" badly enough
            // (mean off by 3 sigma, std half of intended) to flip this test's
            // own result: caught by cross-checking against an INDEPENDENT
            // Python `random.gauss` simulation of the identical quantize/
            // dequantize logic before trusting this Rust generator, the same
            // "verify the generator, not just the formula" discipline this
            // repo's other exhaustive/statistical tests already use.
            ((state >> 32) as u32) as f32 / u32::MAX as f32
        };
        let w: Vec<f32> = (0..n * k)
            .map(|_| {
                let sum: f32 = (0..12).map(|_| next()).sum();
                (sum - 6.0) * 0.02 // approx N(0, 0.02^2)
            })
            .collect();

        let (nf4_packed, nf4_scales) = quantize_weight_lut4(&w, n, k, &NF4_LUT);
        let nf4_deq = dequantize_weight_lut4(&nf4_packed, &nf4_scales, n, k, &NF4_LUT);

        let (q4_packed, q4_scales) = quantize_weight_q4(&w, n, k);
        let q4_deq = crate::int4::dequantize_weight_q4(&q4_packed, &q4_scales, n, k);

        let mse = |deq: &[f32]| -> f64 {
            w.iter().zip(deq.iter()).map(|(&a, &b)| (a as f64 - b as f64).powi(2)).sum::<f64>() / w.len() as f64
        };
        let nf4_mse = mse(&nf4_deq);
        let q4_mse = mse(&q4_deq);
        assert!(
            nf4_mse < q4_mse,
            "NF4's non-uniform codebook should beat Q4's uniform grid on a Gaussian-like weight \
             distribution at the identical 4-bit budget: nf4_mse={nf4_mse:e} q4_mse={q4_mse:e}"
        );
        eprintln!(
            "nf4_beats_q4_on_a_gaussian_like_distribution: q4_mse={q4_mse:e} nf4_mse={nf4_mse:e} \
             ({:.1}% lower)",
            (1.0 - nf4_mse / q4_mse) * 100.0
        );
    }
}

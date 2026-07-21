// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Host-side glue for the Kronos pipeline: per-series normalization (the input
//! contract) and the LSB-first bit↔index packing that is the BSQ tokenizer's
//! discrete interface (the boundary between the GPU `bsq_quantize` kernel and
//! the integer token streams the AR decoder consumes).
//!
//! - [`normalize`] / [`denormalize`] — per-feature z-score over the context time
//!   axis with a `1e-5` floor, then clip to ±`clip` (Kronos `KronosPredictor`).
//! - [`quantized_to_indices`] — a `[T, k]` bipolar BSQ code → two integer token
//!   streams `(s1, s2)`, LSB-first (`bits_to_indices`).
//! - [`indices_to_bipolar`] — the inverse: `(s1, s2)` tokens → `[T, k]` bipolar
//!   values scaled by `1/√k` (`indices_to_bits`), the decoder's input.

/// Per-feature normalization statistics (one entry per feature/column).
#[derive(Clone, Debug, PartialEq)]
pub struct Norm {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

/// Z-score each feature column over the `T` bars, floor std at `1e-5`, then clip
/// to `±clip`. `bars` is `[T, feat]` row-major (each row one OHLCV(+amount) bar).
pub fn normalize(bars: &[f32], t: usize, feat: usize, clip: f32) -> (Norm, Vec<f32>) {
    assert_eq!(bars.len(), t * feat, "bars must be [T, feat]");
    let mut mean = vec![0.0f32; feat];
    let mut std = vec![0.0f32; feat];
    for c in 0..feat {
        let mut s = 0.0f32;
        for r in 0..t {
            s += bars[r * feat + c];
        }
        mean[c] = if t > 0 { s / t as f32 } else { 0.0 };
    }
    for c in 0..feat {
        let mut v = 0.0f32;
        for r in 0..t {
            let d = bars[r * feat + c] - mean[c];
            v += d * d;
        }
        let var = if t > 0 { v / t as f32 } else { 0.0 };
        std[c] = var.sqrt();
    }
    let mut out = vec![0.0f32; t * feat];
    for r in 0..t {
        for c in 0..feat {
            let z = (bars[r * feat + c] - mean[c]) / (std[c] + 1e-5);
            out[r * feat + c] = z.clamp(-clip, clip);
        }
    }
    (Norm { mean, std }, out)
}

/// Invert [`normalize`] (without the clip, which is not invertible): `x*(std+1e-5)
/// + mean` per feature.
pub fn denormalize(norm: &Norm, bars: &[f32], t: usize, feat: usize) -> Vec<f32> {
    assert_eq!(bars.len(), t * feat, "bars must be [T, feat]");
    let mut out = vec![0.0f32; t * feat];
    for r in 0..t {
        for c in 0..feat {
            out[r * feat + c] = bars[r * feat + c] * (norm.std[c] + 1e-5) + norm.mean[c];
        }
    }
    out
}

/// Pack a `[T, k]` bipolar BSQ code (values ≥0 are bit 1) into two integer token
/// streams, LSB-first: `s1` from the first `s1_bits`, `s2` from the next
/// `s2_bits`. `k = s1_bits + s2_bits`.
pub fn quantized_to_indices(
    zq: &[f32],
    t: usize,
    s1_bits: usize,
    s2_bits: usize,
) -> (Vec<u32>, Vec<u32>) {
    let k = s1_bits + s2_bits;
    assert_eq!(zq.len(), t * k, "zq must be [T, k]");
    let pack = |row: &[f32]| -> u32 {
        let mut idx = 0u32;
        for (j, &v) in row.iter().enumerate() {
            if v >= 0.0 {
                idx |= 1 << j; // LSB-first
            }
        }
        idx
    };
    let mut s1 = vec![0u32; t];
    let mut s2 = vec![0u32; t];
    for r in 0..t {
        let base = r * k;
        s1[r] = pack(&zq[base..base + s1_bits]);
        s2[r] = pack(&zq[base + s1_bits..base + k]);
    }
    (s1, s2)
}

/// Unpack two token streams into a `[T, k]` bipolar code scaled by `1/√k` — the
/// inverse of [`quantized_to_indices`], matching the tokenizer's decode input.
pub fn indices_to_bipolar(
    s1: &[u32],
    s2: &[u32],
    s1_bits: usize,
    s2_bits: usize,
) -> Vec<f32> {
    let t = s1.len();
    assert_eq!(s2.len(), t, "s1/s2 length mismatch");
    let k = s1_bits + s2_bits;
    let scale = 1.0 / (k as f32).sqrt();
    let mut out = vec![0.0f32; t * k];
    for r in 0..t {
        let base = r * k;
        for j in 0..s1_bits {
            let bit = (s1[r] >> j) & 1;
            out[base + j] = (bit as f32 * 2.0 - 1.0) * scale;
        }
        for j in 0..s2_bits {
            let bit = (s2[r] >> j) & 1;
            out[base + s1_bits + j] = (bit as f32 * 2.0 - 1.0) * scale;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_zscores_each_feature_and_clips() {
        // 2 features, T=3. feat0 = [1,2,3] mean 2 std sqrt(2/3); feat1 constant 5.
        let bars = vec![1.0, 5.0, 2.0, 5.0, 3.0, 5.0];
        let (n, z) = normalize(&bars, 3, 2, 5.0);
        assert!((n.mean[0] - 2.0).abs() < 1e-6);
        assert!((n.mean[1] - 5.0).abs() < 1e-6);
        // constant feature: std ~0 -> z ~ 0 (divided by 1e-5 but numerator 0)
        assert!(z[1].abs() < 1e-3);
        // feat0 first value normalized: (1-2)/(std+1e-5)
        let s0 = (2.0f32 / 3.0).sqrt();
        assert!((z[0] - (-1.0 / (s0 + 1e-5))).abs() < 1e-4);
    }

    #[test]
    fn normalize_denormalize_roundtrip_without_clip() {
        let bars = vec![10.0, 100.0, 12.0, 110.0, 11.0, 105.0, 13.0, 120.0];
        let (n, z) = normalize(&bars, 4, 2, 100.0); // clip high so nothing clips
        let back = denormalize(&n, &z, 4, 2);
        for i in 0..bars.len() {
            assert!((back[i] - bars[i]).abs() < 1e-2, "roundtrip {i}: {} vs {}", back[i], bars[i]);
        }
    }

    #[test]
    fn clip_bounds_the_normalized_range() {
        let bars = vec![0.0, 0.0, 0.0, 100.0]; // T=4, feat=1, one huge outlier
        let (_, z) = normalize(&bars, 4, 1, 1.5);
        assert!(z.iter().all(|&v| (-1.5..=1.5).contains(&v)), "clipped: {z:?}");
    }

    #[test]
    fn bits_pack_lsb_first_and_roundtrip() {
        // k=4, s1_bits=s2_bits=2. bipolar: >=0 -> bit 1.
        // row: [+, -, +, +] -> s1 bits [1,0] LSB-first = 1; s2 bits [1,1] = 3.
        let zq = vec![0.3, -0.3, 0.3, 0.3];
        let (s1, s2) = quantized_to_indices(&zq, 1, 2, 2);
        assert_eq!(s1, vec![1]);
        assert_eq!(s2, vec![3]);
        // unpack back to bipolar/√4 = ±0.5
        let back = indices_to_bipolar(&s1, &s2, 2, 2);
        assert_eq!(back, vec![0.5, -0.5, 0.5, 0.5]);
    }

    #[test]
    fn index_bit_ordering_is_stable_across_pack_unpack() {
        // full-range roundtrip: every 10-bit index maps to bits and back.
        let s1: Vec<u32> = vec![0, 1, 2, 511, 1023];
        let s2: Vec<u32> = vec![1023, 512, 5, 0, 42];
        let bip = indices_to_bipolar(&s1, &s2, 10, 10);
        let (r1, r2) = quantized_to_indices(&bip, s1.len(), 10, 10);
        assert_eq!(r1, s1);
        assert_eq!(r2, s2);
    }
}

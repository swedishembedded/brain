// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded low-precision training for its
// clients. If your team needs expertise in 8-bit optimiser state and quantised
// full fine-tuning then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Tile codecs for optimiser state: f32, bf16 and blockwise 8-bit, with
//! optional stochastic rounding. Every codec works on a TILE (a contiguous
//! element range that starts on a block boundary), so state of any size is
//! decoded and re-encoded in bounded workspace.
//!
//! # 8-bit maps
//!
//! Each block of `block` elements shares one f32 scale, its absolute maximum.
//! Inside a block the value `x/scale` is companded before rounding to a code,
//! because moments are heavy-tailed and a linear map spends most codes on the
//! few large entries:
//!
//! * [`SlotKind::Signed`] (first moment, momentum, master): sign plus a 7-bit
//!   magnitude on a square-root map, `x = sign * (q/127)^2 * scale`.
//! * [`SlotKind::NonNegative`] (second moment): 8-bit magnitude on a
//!   fourth-root map, `x = (q/255)^4 * scale`. A strictly positive input never
//!   encodes to code 0: a second moment decoded as exactly zero turns the
//!   update into `m / eps`, the classic 8-bit-Adam instability.
//!
//! Both are exactly zero for an all-zero block (scale 0, codes 0), so a
//! zero-filled file is a valid empty state.
//!
//! # Rounding
//!
//! Stochastic rounding draws its randomness from a counter-based hash of
//! `(key, absolute element index)` - not from a stream - so the result does
//! not depend on tiling, thread count or whether the run was resumed.

use paramstore::precision::StateFormat;

/// Sign domain of a state slot; selects the 8-bit code map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotKind {
    Signed,
    NonNegative,
}

/// How a re-encode rounds. `key` should identify (run seed, step, tensor,
/// slot); the element index is mixed in per element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rounding {
    pub stochastic: bool,
    pub key: u64,
}

impl Rounding {
    pub const NEAREST: Rounding = Rounding { stochastic: false, key: 0 };
}

/// splitmix64 finaliser.
#[inline]
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Combine identifying fields into one rounding key.
pub fn rounding_key(seed: u64, step: u32, tensor: usize, slot: usize) -> u64 {
    mix(mix(mix(seed ^ step as u64) ^ tensor as u64) ^ slot as u64)
}

/// Uniform in `[0, 1)` for element `idx` under `key`.
#[inline]
fn unit(key: u64, idx: u64) -> f32 {
    ((mix(key ^ mix(idx)) >> 40) as f32) * (1.0 / (1u64 << 24) as f32)
}

/// f32 -> bf16 bits. Nearest-even, or stochastic with `u` in `[0,1)`.
#[inline]
pub fn f32_to_bf16(x: f32, u: Option<f32>) -> u16 {
    let b = x.to_bits();
    if !x.is_finite() {
        return (b >> 16) as u16 | if x.is_nan() { 0x0040 } else { 0 };
    }
    match u {
        None => {
            let lsb = (b >> 16) & 1;
            (b.wrapping_add(0x7FFF + lsb) >> 16) as u16
        }
        Some(u) => {
            // Carry into the kept bits with probability (dropped / 2^16).
            let r = (u * 65536.0) as u32 & 0xFFFF;
            (b.wrapping_add(r) >> 16) as u16
        }
    }
}

#[inline]
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Codec of one state slot: a format plus the slot's sign domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateCodec {
    pub format: StateFormat,
    pub kind: SlotKind,
}

impl StateCodec {
    /// Bytes of the code region for `n` elements.
    pub fn data_bytes(&self, n: usize) -> usize {
        match self.format {
            StateFormat::F32 => 4 * n,
            StateFormat::Bf16 => 2 * n,
            StateFormat::Int8Block { .. } => n,
        }
    }

    /// Bytes of the scale region for `n` elements (0 for unblocked formats).
    pub fn scale_bytes(&self, n: usize) -> usize {
        match self.format {
            StateFormat::Int8Block { block } => 4 * n.div_ceil(block as usize),
            _ => 0,
        }
    }

    /// Decode a tile. `data` and `scales` are the tile's own regions; `out.len()`
    /// is the element count.
    pub fn decode(&self, data: &[u8], scales: &[u8], out: &mut [f32]) {
        match self.format {
            StateFormat::F32 => {
                for (o, c) in out.iter_mut().zip(data.as_chunks::<4>().0) {
                    *o = f32::from_le_bytes(*c);
                }
            }
            StateFormat::Bf16 => {
                for (o, c) in out.iter_mut().zip(data.as_chunks::<2>().0) {
                    *o = bf16_to_f32(u16::from_le_bytes(*c));
                }
            }
            StateFormat::Int8Block { block } => {
                for (bi, chunk) in out.chunks_mut(block as usize).enumerate() {
                    let s = f32::from_le_bytes(scales[4 * bi..4 * bi + 4].try_into().unwrap());
                    let codes = &data[bi * block as usize..];
                    for (o, &q) in chunk.iter_mut().zip(codes) {
                        *o = decode_code(self.kind, q, s);
                    }
                }
            }
        }
    }

    /// Encode a tile whose first element has absolute index `base` in its
    /// tensor (the rounding randomness is keyed on it).
    pub fn encode(&self, vals: &[f32], data: &mut [u8], scales: &mut [u8], rnd: Rounding, base: u64) {
        let u = |i: usize| rnd.stochastic.then(|| unit(rnd.key, base + i as u64));
        match self.format {
            StateFormat::F32 => {
                for (c, v) in data.as_chunks_mut::<4>().0.iter_mut().zip(vals) {
                    *c = v.to_le_bytes();
                }
            }
            StateFormat::Bf16 => {
                for (i, (c, v)) in data.as_chunks_mut::<2>().0.iter_mut().zip(vals).enumerate() {
                    *c = f32_to_bf16(*v, u(i)).to_le_bytes();
                }
            }
            StateFormat::Int8Block { block } => {
                let block = block as usize;
                for (bi, chunk) in vals.chunks(block).enumerate() {
                    let absmax = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
                    scales[4 * bi..4 * bi + 4].copy_from_slice(&absmax.to_le_bytes());
                    for (j, v) in chunk.iter().enumerate() {
                        data[bi * block + j] = encode_code(self.kind, *v, absmax, u(bi * block + j));
                    }
                }
            }
        }
    }
}

/// Code -> value for one 8-bit element.
#[inline]
pub fn decode_code(kind: SlotKind, q: u8, scale: f32) -> f32 {
    match kind {
        SlotKind::Signed => {
            let c = q as i8;
            let r = c.unsigned_abs() as f32 / 127.0;
            let m = r * r * scale;
            if c < 0 {
                -m
            } else {
                m
            }
        }
        SlotKind::NonNegative => {
            let r = q as f32 / 255.0;
            let r2 = r * r;
            r2 * r2 * scale
        }
    }
}

#[inline]
fn round_code(x: f32, u: Option<f32>) -> f32 {
    match u {
        None => (x + 0.5).floor(),
        Some(u) => (x + u).floor(),
    }
}

/// Value -> code for one 8-bit element of a block with maximum `absmax`.
#[inline]
pub fn encode_code(kind: SlotKind, v: f32, absmax: f32, u: Option<f32>) -> u8 {
    if absmax <= 0.0 || !v.is_finite() {
        return 0;
    }
    match kind {
        SlotKind::Signed => {
            let q = round_code((v.abs() / absmax).min(1.0).sqrt() * 127.0, u).clamp(0.0, 127.0) as i8;
            (if v < 0.0 { -q } else { q }) as u8
        }
        SlotKind::NonNegative => {
            if v <= 0.0 {
                return 0;
            }
            let q = round_code((v / absmax).min(1.0).sqrt().sqrt() * 255.0, u).clamp(1.0, 255.0);
            q as u8
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(c: StateCodec, vals: &[f32], rnd: Rounding) -> Vec<f32> {
        let mut data = vec![0u8; c.data_bytes(vals.len())];
        let mut sc = vec![0u8; c.scale_bytes(vals.len())];
        c.encode(vals, &mut data, &mut sc, rnd, 0);
        let mut out = vec![0f32; vals.len()];
        c.decode(&data, &sc, &mut out);
        out
    }

    /// Spec: f32 is lossless, bf16 is the nearest bf16, an all-zero state
    /// decodes to zero in every format.
    #[test]
    fn f32_exact_bf16_nearest_zero_is_zero() {
        let vals = [1.0f32, -2.5, 1e-20, std::f32::consts::PI];
        let c32 = StateCodec { format: StateFormat::F32, kind: SlotKind::Signed };
        assert_eq!(roundtrip(c32, &vals, Rounding::NEAREST), vals);
        let c16 = StateCodec { format: StateFormat::Bf16, kind: SlotKind::Signed };
        for (a, b) in vals.iter().zip(roundtrip(c16, &vals, Rounding::NEAREST)) {
            assert!((a - b).abs() <= a.abs() * 2f32.powi(-8), "{a} vs {b}");
        }
        let c8 = StateCodec { format: StateFormat::Int8Block { block: 4 }, kind: SlotKind::NonNegative };
        assert_eq!(roundtrip(c8, &[0.0; 8], Rounding::NEAREST), [0.0; 8]);
    }

    /// Spec: int8 codes carry a per-block scale, so a block of tiny values
    /// beside a block of huge ones keeps its own relative precision (a
    /// tensor-wide linear scale would flush the small block to zero).
    #[test]
    fn int8_is_blockwise_not_tensor_wide() {
        let c = StateCodec { format: StateFormat::Int8Block { block: 8 }, kind: SlotKind::Signed };
        let mut vals = vec![1000.0f32; 8];
        vals.extend((0..8).map(|i| 1e-3 * (i as f32 - 3.5)));
        let out = roundtrip(c, &vals, Rounding::NEAREST);
        for i in 8..16 {
            let rel = (out[i] - vals[i]).abs() / vals[i].abs();
            assert!(rel < 0.05, "small block element {i}: {} vs {}", out[i], vals[i]);
        }
    }

    /// Spec: a strictly positive second moment never decodes to zero, however
    /// small relative to its block maximum.
    #[test]
    fn nonnegative_never_collapses_positive_to_zero() {
        let c = StateCodec { format: StateFormat::Int8Block { block: 4 }, kind: SlotKind::NonNegative };
        let out = roundtrip(c, &[1.0, 1e-30, 1e-12, 0.5], Rounding::NEAREST);
        assert!(out.iter().all(|&v| v > 0.0), "{out:?}");
        assert!((out[0] - 1.0).abs() < 0.02 && (out[3] - 0.5).abs() < 0.02);
    }

    /// Spec: stochastic rounding is unbiased - re-encoding a value that sits
    /// between two codes many times averages to the value, where nearest
    /// rounding is stuck on one side. This is what lets a 0.1% EMA decay
    /// register in an 8-bit moment.
    #[test]
    fn stochastic_rounding_is_unbiased_and_tiling_independent() {
        let c = StateCodec { format: StateFormat::Int8Block { block: 64 }, kind: SlotKind::NonNegative };
        let mut vals = vec![0.5f32; 64];
        vals[0] = 1.0; // pin the block scale
        let target = 0.5003f32;
        vals[1] = target;
        let (mut sum, n) = (0f64, 4000u64);
        for k in 0..n {
            let mut data = vec![0u8; 64];
            let mut sc = vec![0u8; 4];
            c.encode(&vals, &mut data, &mut sc, Rounding { stochastic: true, key: k }, 0);
            let mut out = vec![0f32; 64];
            c.decode(&data, &sc, &mut out);
            sum += out[1] as f64;
        }
        let mean = sum / n as f64;
        assert!((mean - target as f64).abs() < 3e-4, "mean {mean}");
        let nearest = roundtrip(c, &vals, Rounding::NEAREST)[1];
        assert!((nearest - target).abs() > 2e-4, "nearest {nearest} should be stuck off-target");

        // Same key, same absolute indices -> same bits whatever the tiling.
        let key = Rounding { stochastic: true, key: 7 };
        let cb = StateCodec { format: StateFormat::Bf16, kind: SlotKind::Signed };
        let v: Vec<f32> = (0..32).map(|i| 1.0 + i as f32 * 1e-3).collect();
        let (mut whole, mut parts) = (vec![0u8; 64], vec![0u8; 64]);
        cb.encode(&v, &mut whole, &mut [], key, 100);
        cb.encode(&v[..16], &mut parts[..32], &mut [], key, 100);
        cb.encode(&v[16..], &mut parts[32..], &mut [], key, 116);
        assert_eq!(whole, parts);
    }

}

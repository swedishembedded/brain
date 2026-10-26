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

/// 32-bit integer mixer. The rounding randomness is defined on 32-bit words
/// so the device kernels (`adamw_q8.wgsl`, which has no 64-bit integers) draw
/// bit-identical numbers.
#[inline]
pub fn mix32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

/// Combine identifying fields into one rounding key (low word from the seed
/// and step, high word from the seed and the tensor/slot).
pub fn rounding_key(seed: u64, step: u32, tensor: usize, slot: usize) -> u64 {
    let lo = mix32(seed as u32 ^ mix32(step));
    let ts = (((tensor as u32) << 8) | (slot as u32 & 0xff)).wrapping_add(0x9E37_79B9);
    let hi = mix32((seed >> 32) as u32 ^ mix32(ts));
    ((hi as u64) << 32) | lo as u64
}

/// Uniform in `[0, 1)` for element `idx` under `key`.
#[inline]
fn unit(key: u64, idx: u64) -> f32 {
    let (k0, k1) = (key as u32, (key >> 32) as u32);
    let h = mix32(mix32(k0 ^ idx as u32) ^ k1 ^ mix32((idx >> 32) as u32));
    ((h >> 8) as f32) * (1.0 / (1u64 << 24) as f32)
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
            StateFormat::Int8Block { .. } | StateFormat::Log8Block { .. } => n,
        }
    }

    /// Bytes of the scale region for `n` elements (0 for unblocked formats).
    pub fn scale_bytes(&self, n: usize) -> usize {
        match self.format {
            StateFormat::Int8Block { block } | StateFormat::Log8Block { block } => 4 * n.div_ceil(block as usize),
            _ => 0,
        }
    }

    /// Decode a tile. `data` and `scales` are the tile's own regions; `out.len()`
    /// is the element count. A large tile of a block or bf16 format is cut into whole blocks and decoded on scoped
    /// threads; the bits are those of the serial decode.
    pub fn decode(&self, data: &[u8], scales: &[u8], out: &mut [f32]) {
        let (pieces, piece) = self.split(out.len());
        if pieces <= 1 {
            return self.decode_serial(data, scales, out);
        }
        crate::par::chunks_mut(out, piece, |i, chunk| {
            let e0 = i * piece;
            let (d0, s0) = (self.data_bytes(e0), self.scale_bytes(e0));
            let (d1, s1) = (d0 + self.data_bytes(chunk.len()), s0 + self.scale_bytes(chunk.len()));
            self.decode_serial(&data[d0..d1], &scales[s0..s1], chunk);
        });
    }

    /// Pieces (and their length in elements, a whole number of blocks) a tile of `n` elements is worked in:
    /// one piece unless the tile is large and the format costs more than a copy.
    fn split(&self, n: usize) -> (usize, usize) {
        let block = self.format.block().max(2);
        let threads = crate::par::threads().min(n / crate::par::MIN_PIECE_ELEMS);
        if threads <= 1 {
            return (1, n.max(1));
        }
        let piece = n.div_ceil(threads).div_ceil(block) * block;
        (n.div_ceil(piece), piece)
    }

    pub(crate) fn decode_serial(&self, data: &[u8], scales: &[u8], out: &mut [f32]) {
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
            StateFormat::Int8Block { block } | StateFormat::Log8Block { block } => {
                let log = matches!(self.format, StateFormat::Log8Block { .. });
                for (bi, chunk) in out.chunks_mut(block as usize).enumerate() {
                    let s = f32::from_le_bytes(scales[4 * bi..4 * bi + 4].try_into().unwrap());
                    let codes = &data[bi * block as usize..];
                    for (o, &q) in chunk.iter_mut().zip(codes) {
                        *o = if log { decode_log(self.kind, q, s) } else { decode_code(self.kind, q, s) };
                    }
                }
            }
        }
    }

    /// Encode a tile whose first element has absolute index `base` in its
    /// tensor (the rounding randomness is keyed on it). Large tiles are encoded in whole blocks on scoped threads;
    /// the rounding draws are keyed on absolute element indices, so the bits do not depend on the cut.
    pub fn encode(&self, vals: &[f32], data: &mut [u8], scales: &mut [u8], rnd: Rounding, base: u64) {
        let (pieces, piece) = self.split(vals.len());
        if pieces <= 1 {
            return self.encode_serial(vals, data, scales, rnd, base);
        }
        let encode_piece = |i: usize, d: &mut [u8], sc: &mut [u8]| {
            let from = i * piece;
            self.encode_serial(&vals[from..(from + piece).min(vals.len())], d, sc, rnd, base + from as u64);
        };
        match self.scale_bytes(piece) {
            0 => crate::par::chunks_mut(data, self.data_bytes(piece), |i, d| encode_piece(i, d, &mut [])),
            sb => crate::par::chunks2_mut(data, self.data_bytes(piece), scales, sb, encode_piece),
        }
    }

    pub(crate) fn encode_serial(&self, vals: &[f32], data: &mut [u8], scales: &mut [u8], rnd: Rounding, base: u64) {
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
            StateFormat::Int8Block { block } | StateFormat::Log8Block { block } => {
                let log = matches!(self.format, StateFormat::Log8Block { .. });
                let block = block as usize;
                for (bi, chunk) in vals.chunks(block).enumerate() {
                    let absmax = chunk.iter().fold(0f32, |m, v| m.max(v.abs()));
                    scales[4 * bi..4 * bi + 4].copy_from_slice(&absmax.to_le_bytes());
                    let inv = if absmax > 0.0 { 1.0 / absmax } else { 0.0 };
                    for (j, v) in chunk.iter().enumerate() {
                        let rnd = u(bi * block + j);
                        data[bi * block + j] = if log { encode_log(self.kind, *v, inv, rnd) } else { encode_code(self.kind, *v, inv, rnd) };
                    }
                }
            }
        }
    }
}

/// f32 nearest to 1/127 and 1/255, spelled out so the WGSL kernels use the identical constants.
pub const INV_127: f32 = 0.007_874_016;
pub const INV_255: f32 = 0.003_921_569;

/// Code -> value for one 8-bit element.
#[inline]
pub fn decode_code(kind: SlotKind, q: u8, scale: f32) -> f32 {
    match kind {
        SlotKind::Signed => {
            let c = q as i8;
            let r = c.unsigned_abs() as f32 * INV_127;
            let m = r * r * scale;
            if c < 0 {
                -m
            } else {
                m
            }
        }
        SlotKind::NonNegative => {
            let r = q as f32 * INV_255;
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

/// Value -> code for one 8-bit element of a block whose maximum has reciprocal
/// `inv` (0 for an all-zero block). A reciprocal multiply rather than a divide
/// per element: the device kernels do the same, bit for bit.
#[inline]
pub fn encode_code(kind: SlotKind, v: f32, inv: f32, u: Option<f32>) -> u8 {
    if inv <= 0.0 || !v.is_finite() {
        return 0;
    }
    match kind {
        SlotKind::Signed => {
            let q = round_code((v.abs() * inv).min(1.0).sqrt() * 127.0, u).clamp(0.0, 127.0) as i8;
            (if v < 0.0 { -q } else { q }) as u8
        }
        SlotKind::NonNegative => {
            if v <= 0.0 {
                return 0;
            }
            let q = round_code((v * inv).min(1.0).sqrt().sqrt() * 255.0, u).clamp(1.0, 255.0);
            q as u8
        }
    }
}

/// Log-code levels per octave: signed codes (7 magnitude bits) take 8 steps an octave, non-negative codes
/// (8 bits) 16, so both span about 16 octaves below the block maximum.
const LOG_STEPS_SIGNED: f32 = 8.0;
const LOG_STEPS_NONNEG: f32 = 16.0;

/// Code -> value for one logarithmic 8-bit element: `scale * 2^((q - top) / steps)`, with the lowest signed
/// code (magnitude 0) meaning zero.
#[inline]
pub fn decode_log(kind: SlotKind, q: u8, scale: f32) -> f32 {
    match kind {
        SlotKind::Signed => {
            let c = q as i8;
            let m = c.unsigned_abs();
            if m == 0 {
                return 0.0;
            }
            let v = scale * ((m as f32 - 127.0) / LOG_STEPS_SIGNED).exp2();
            if c < 0 {
                -v
            } else {
                v
            }
        }
        SlotKind::NonNegative => {
            if q == 0 {
                return 0.0;
            }
            scale * ((q as f32 - 255.0) / LOG_STEPS_NONNEG).exp2()
        }
    }
}

/// Value -> code for one logarithmic element of a block whose maximum has reciprocal `inv`; rounding is in the
/// log domain (nearest, or stochastic with `u`). Values below the lowest level snap to it, never to zero, for
/// a non-negative slot (a positive second moment must not vanish); a signed slot flushes them to zero.
#[inline]
pub fn encode_log(kind: SlotKind, v: f32, inv: f32, u: Option<f32>) -> u8 {
    if inv <= 0.0 || !v.is_finite() {
        return 0;
    }
    let (top, steps) = match kind {
        SlotKind::Signed => (127.0, LOG_STEPS_SIGNED),
        SlotKind::NonNegative => (255.0, LOG_STEPS_NONNEG),
    };
    let a = (v.abs() * inv).min(1.0);
    if a <= 0.0 || (kind == SlotKind::NonNegative && v < 0.0) {
        return 0;
    }
    let level = round_code(top + a.log2() * steps, u);
    match kind {
        SlotKind::Signed => {
            let m = level.clamp(0.0, 127.0) as i8;
            (if v < 0.0 { -m } else { m }) as u8
        }
        SlotKind::NonNegative => level.clamp(1.0, 255.0) as u8,
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

    /// Spec: a logarithmic block keeps a constant RELATIVE step over its range: elements from the block maximum
    /// down thirteen octaves round-trip within a few percent each, where the int8 map loses the small ones; a
    /// positive second moment never decodes to zero.
    #[test]
    fn log8_has_constant_relative_precision_over_many_octaves() {
        for (kind, bound) in [(SlotKind::NonNegative, 0.03f32), (SlotKind::Signed, 0.05)] {
            let c = StateCodec { format: StateFormat::Log8Block { block: 64 }, kind };
            let vals: Vec<f32> = (0..64).map(|i| (-(i as f32) * 13.0 / 63.0).exp2()).collect();
            let out = roundtrip(c, &vals, Rounding::NEAREST);
            for (a, b) in vals.iter().zip(&out) {
                assert!(((a - b) / a).abs() < bound, "{kind:?}: {a} -> {b}");
            }
            let i8c = StateCodec { format: StateFormat::Int8Block { block: 64 }, kind };
            let i8out = roundtrip(i8c, &vals, Rounding::NEAREST);
            let worst_i8 = vals.iter().zip(&i8out).map(|(a, b)| ((a - b) / a).abs()).fold(0f32, f32::max);
            assert!(worst_i8 > bound, "{kind:?}: the int8 map is the coarser one here ({worst_i8})");
        }
        let c = StateCodec { format: StateFormat::Log8Block { block: 4 }, kind: SlotKind::NonNegative };
        assert!(roundtrip(c, &[1.0, 1e-30, 1e-12, 0.5], Rounding::NEAREST).iter().all(|&v| v > 0.0));
        assert_eq!(roundtrip(c, &[0.0; 4], Rounding::NEAREST), [0.0; 4]);
    }

    /// Spec: logarithmic stochastic rounding is unbiased in the log domain: a value between two levels is
    /// reached on average, which nearest rounding cannot do.
    #[test]
    fn log8_stochastic_rounding_averages_to_the_value_in_the_log_domain() {
        let c = StateCodec { format: StateFormat::Log8Block { block: 4 }, kind: SlotKind::NonNegative };
        let vals = [1.0f32, 0.5 * 1.021, 0.5, 0.25];
        let target = vals[1].log2() as f64;
        let n = 4000u64;
        let mut sum = 0f64;
        for k in 0..n {
            let mut data = vec![0u8; 4];
            let mut sc = vec![0u8; 4];
            c.encode(&vals, &mut data, &mut sc, Rounding { stochastic: true, key: k }, 0);
            let mut out = vec![0f32; 4];
            c.decode(&data, &sc, &mut out);
            sum += (out[1] as f64).log2();
        }
        assert!((sum / n as f64 - target).abs() < 2e-3, "mean log2 {} vs {target}", sum / n as f64);
    }

    /// Spec: a large tile is encoded and decoded in pieces on several threads and gives exactly the bits the serial
    /// loop gives, stochastic rounding included (its draws are keyed on absolute element indices).
    #[test]
    fn large_tiles_are_worked_in_pieces_with_the_serial_bits() {
        let n = 300_011usize;
        let vals: Vec<f32> = (0..n).map(|i| (((i * 2_654_435_761) % 10_007) as f32 / 10_007.0 - 0.3) * 4.0).collect();
        for format in [StateFormat::F32, StateFormat::Bf16, StateFormat::Int8Block { block: 64 }, StateFormat::Int8Block { block: 256 }, StateFormat::Log8Block { block: 64 }] {
            for kind in [SlotKind::Signed, SlotKind::NonNegative] {
                let c = StateCodec { format, kind };
                let v: Vec<f32> = if kind == SlotKind::NonNegative { vals.iter().map(|x| x.abs()).collect() } else { vals.clone() };
                for rnd in [Rounding::NEAREST, Rounding { stochastic: true, key: 99 }] {
                    let (mut d1, mut s1) = (vec![0u8; c.data_bytes(n)], vec![0u8; c.scale_bytes(n)]);
                    let (mut d2, mut s2) = (d1.clone(), s1.clone());
                    c.encode(&v, &mut d1, &mut s1, rnd, 1234);
                    c.encode_serial(&v, &mut d2, &mut s2, rnd, 1234);
                    assert!(d1 == d2 && s1 == s2, "{format:?} {kind:?} {rnd:?}: encode");
                    let (mut o1, mut o2) = (vec![0f32; n], vec![0f32; n]);
                    c.decode(&d1, &s1, &mut o1);
                    c.decode_serial(&d1, &s1, &mut o2);
                    assert!(o1 == o2, "{format:?} {kind:?}: decode");
                }
            }
        }
    }
}

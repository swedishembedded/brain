// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded loading and training of quantized models for its clients. If your
// team needs expertise in low-precision model storage then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Row-level quantizers for the formats a load can target (f32, f16, bf16,
//! block-scaled int8, block-scaled NF4), and the [`encode`] tile entry point.
//!
//! The row primitives are the single implementation of brain's packed storage
//! conventions (4 int8 or 8 NF4 codes per little-endian `u32`, one f32 scale per
//! block); `model::int8` and `model::lut4` call them, so a converting load and a
//! runtime quantizer cannot produce different bytes for the same input.

use std::ops::Range;

use super::{TileBuf, NF4_CODEBOOK};
use crate::packed::encoding::{Codebook, Element, FloatFormat, QuantEncoding, ScaleGranularity, TensorDescriptor};

/// One row's block scales for symmetric int8: `max|block| / 127`, floored so an
/// all-zero block cannot divide by zero. The fold is left to right: with a NaN
/// present the order is observable, and every caller must agree on it.
#[inline]
pub fn int8_block_scales(row: &[f32], group: usize, out: &mut [f32]) {
    debug_assert_eq!(row.len(), out.len() * group);
    for (g, s) in out.iter_mut().enumerate() {
        let blk = &row[g * group..(g + 1) * group];
        *s = blk.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-8) / 127.0;
    }
}

/// Pack one row to int8 words (4 lanes per word, low byte first) given its
/// block scales. `group` must be a multiple of 4.
#[inline]
pub fn int8_pack_row(row: &[f32], group: usize, scales: &[f32], out: &mut [u32]) {
    let words_per_group = group / 4;
    debug_assert_eq!(out.len(), scales.len() * words_per_group);
    for (g, word_out) in out.iter_mut().enumerate() {
        let inv = 1.0 / scales[g / words_per_group];
        let mut word = 0u32;
        for b in 0..4 {
            let q = (row[g * 4 + b] * inv).round().clamp(-127.0, 127.0) as i32;
            word |= ((q as u8) as u32) << (8 * b);
        }
        *word_out = word;
    }
}

/// Quantize one row to a 16-entry codebook: per block scale
/// `max|block| / max|lut|`, each element the nearest codebook entry, eight
/// 4-bit codes per word (code `b` at bits `4b`). `group` must be a multiple of 8.
pub fn codebook4_quantize_row(row: &[f32], group: usize, lut: &[f32; 16], words: &mut [u32], scales: &mut [f32]) {
    let lut_absmax = lut.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-8);
    let words_per_group = group / 8;
    for (g, s) in scales.iter_mut().enumerate() {
        let amax = row[g * group..(g + 1) * group].iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-8);
        *s = amax / lut_absmax;
    }
    for (g, word_out) in words.iter_mut().enumerate() {
        let inv = 1.0 / scales[g / words_per_group];
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
}

/// How [`encode`] produces an encoding.
pub(super) enum EncodeKind {
    Float(FloatFormat),
    Int8Block(usize),
    Nf4Block(usize),
    /// Unscaled FP8.
    Fp8(FloatFormat),
    /// FP8 with one bf16/f32 scale per `rows x cols` tile.
    Fp8Tiled { fmt: FloatFormat, rows: usize, cols: usize, scale: FloatFormat },
    /// E2M1 with an E4M3 scale per 16 and an f32 scale per tensor.
    Nvfp4,
    /// E2M1 with an E8M0 scale per 32.
    Mxfp4,
}

pub(super) fn encode_kind(enc: &QuantEncoding) -> Option<EncodeKind> {
    if *enc == QuantEncoding::nvfp4() {
        return Some(EncodeKind::Nvfp4);
    }
    if *enc == QuantEncoding::mxfp4() {
        return Some(EncodeKind::Mxfp4);
    }
    match (enc.element, enc.scales.as_slice()) {
        (Element::Float(f @ (FloatFormat::E4M3Fn | FloatFormat::E5M2)), []) => return Some(EncodeKind::Fp8(f)),
        (Element::Float(f @ (FloatFormat::E4M3Fn | FloatFormat::E5M2)), [l]) => {
            return match (l.granularity, l.format) {
                (ScaleGranularity::Tile2d { rows, cols }, s @ (FloatFormat::Bf16 | FloatFormat::F32)) => Some(EncodeKind::Fp8Tiled { fmt: f, rows: rows as usize, cols: cols as usize, scale: s }),
                _ => None,
            }
        }
        _ => {}
    }
    let block = match enc.scales.as_slice() {
        [] => None,
        [l] if l.format == FloatFormat::F32 => match l.granularity {
            ScaleGranularity::Block(n) => Some(n as usize),
            _ => return None,
        },
        _ => return None,
    };
    match (enc.element, block) {
        (Element::Float(f @ (FloatFormat::F32 | FloatFormat::F16 | FloatFormat::Bf16)), None) => Some(EncodeKind::Float(f)),
        (Element::Int { bits: 8, signed: true }, Some(n)) if n.is_multiple_of(4) => Some(EncodeKind::Int8Block(n)),
        (Element::Codebook(Codebook::Nf4), Some(n)) if n.is_multiple_of(8) => Some(EncodeKind::Nf4Block(n)),
        _ => None,
    }
}

fn words_to_le(words: &[u32], out: &mut Vec<u8>) {
    out.clear();
    out.extend(words.iter().flat_map(|w| w.to_le_bytes()));
}

fn f32s_to_le(vals: &[f32], out: &mut Vec<u8>) {
    fill_encoded::<4>(vals, out, f32::to_le_bytes);
}

/// `v` as bf16 bits, rounded to nearest even; NaN stays NaN (quiet, payload kept). The bits of `half::bf16::from_f32`
/// in a form without a branch per element, so a tile of values narrows with vector instructions.
#[inline]
fn bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    if v.is_nan() {
        (b >> 16) as u16 | 0x0040
    } else {
        (b.wrapping_add(0x7FFF + ((b >> 16) & 1)) >> 16) as u16
    }
}

/// Elements encoded per parallel work item.
#[cfg(not(target_arch = "wasm32"))]
const ENCODE_CHUNK: usize = 1 << 16;

/// Encode every value of `vals` to a fixed-width little-endian code, replacing `out`. Elements are independent, so
/// large tiles are cut into contiguous chunks run on the CPU scheduler's pool; the bytes are the serial loop's.
fn fill_encoded<const W: usize>(vals: &[f32], out: &mut Vec<u8>, encode: impl Fn(f32) -> [u8; W] + Sync) {
    // Every byte is written below; see `fill_decoded` for why a fresh buffer is zeroed pages and not a `resize`.
    if out.len() != vals.len() * W {
        *out = vec![0u8; vals.len() * W];
    }
    let codes = out.as_chunks_mut::<W>().0;
    #[cfg(not(target_arch = "wasm32"))]
    backend_cpu::par::chunks_mut(codes, ENCODE_CHUNK, |c, dst| {
        for (d, &v) in dst.iter_mut().zip(&vals[c * ENCODE_CHUNK..]) {
            *d = encode(v);
        }
    });
    #[cfg(target_arch = "wasm32")]
    for (d, &v) in codes.iter_mut().zip(vals) {
        *d = encode(v);
    }
}

/// Encode tile `elems` (`values`, row-major) of `desc` into `out`.
pub(super) fn encode(desc: &TensorDescriptor, elems: &Range<u64>, values: &[f32], ctx: &super::EncodeCtx, out: &mut TileBuf) -> Result<(), String> {
    let kind = encode_kind(&desc.encoding).ok_or_else(|| format!("{}: cannot encode {}", desc.name, desc.encoding))?;
    if values.len() as u64 != elems.end - elems.start {
        return Err(format!("{}: {} values for tile {elems:?}", desc.name, values.len()));
    }
    let inner = *desc.shape.last().unwrap_or(&1) as usize;
    if let EncodeKind::Int8Block(n) | EncodeKind::Nf4Block(n) = kind {
        // Block scales are per row, so a tile must be whole rows (or one whole row's blocks).
        if !(elems.start as usize).is_multiple_of(n) || !values.len().is_multiple_of(n) || !inner.is_multiple_of(n) {
            return Err(format!("{}: tile {elems:?} is not aligned to {n}-element blocks", desc.name));
        }
    }
    if let EncodeKind::Fp8Tiled { rows, .. } = kind {
        let band = rows * inner;
        if !(elems.start as usize).is_multiple_of(band) || !(values.len().is_multiple_of(band) || elems.end == desc.numel()) {
            return Err(format!("{}: tile {elems:?} is not aligned to whole {rows}-row scale bands", desc.name));
        }
    }
    out.scales.clear();
    match kind {
        EncodeKind::Fp8(fmt) => {
            let grid = grid(fmt);
            fill_encoded::<1>(values, &mut out.payload, |v| [nearest(grid, v)]);
        }
        EncodeKind::Fp8Tiled { fmt, rows, cols, scale } => encode_fp8_tiled(values, inner, fmt, rows, cols, scale, out),
        EncodeKind::Nvfp4 => encode_nvfp4(values, ctx.tensor_absmax, out),
        EncodeKind::Mxfp4 => encode_mxfp4(values, out),
        EncodeKind::Float(FloatFormat::F32) => f32s_to_le(values, &mut out.payload),
        EncodeKind::Float(FloatFormat::F16) => fill_encoded::<2>(values, &mut out.payload, |v| half::f16::from_f32(v).to_le_bytes()),
        EncodeKind::Float(_) => fill_encoded::<2>(values, &mut out.payload, |v| bf16_bits(v).to_le_bytes()),
        EncodeKind::Int8Block(n) => {
            let mut scales = vec![0f32; values.len() / n];
            int8_block_scales(values, n, &mut scales);
            let mut words = vec![0u32; values.len() / 4];
            int8_pack_row(values, n, &scales, &mut words);
            words_to_le(&words, &mut out.payload);
            out.scales.push(Vec::new());
            f32s_to_le(&scales, &mut out.scales[0]);
        }
        EncodeKind::Nf4Block(n) => {
            let mut scales = vec![0f32; values.len() / n];
            let mut words = vec![0u32; values.len() / 8];
            codebook4_quantize_row(values, n, &NF4_CODEBOOK, &mut words, &mut scales);
            words_to_le(&words, &mut out.payload);
            out.scales.push(Vec::new());
            f32s_to_le(&scales, &mut out.scales[0]);
        }
    }
    Ok(())
}

/// The finite non-negative codes of a float format with their values, ascending, for nearest-value
/// encoding. Index == value order, so a binary search finds the neighbours.
struct Grid {
    /// `(value, code)`, ascending by value.
    mags: Vec<(f32, u8)>,
    /// The code's sign bit.
    sign: u8,
}

fn grid(fmt: FloatFormat) -> &'static Grid {
    use std::sync::OnceLock;
    static E4M3: OnceLock<Grid> = OnceLock::new();
    static E5M2: OnceLock<Grid> = OnceLock::new();
    static E2M1: OnceLock<Grid> = OnceLock::new();
    let build = |codes: u16, sign: u8| {
        let mut mags: Vec<(f32, u8)> = (0..codes as u8).map(|c| (super::decode_float_code(fmt, c as u32), c)).filter(|(v, _)| v.is_finite()).collect();
        mags.sort_by(|a, b| a.0.total_cmp(&b.0));
        Grid { mags, sign }
    };
    match fmt {
        FloatFormat::E4M3Fn => E4M3.get_or_init(|| build(0x80, 0x80)),
        FloatFormat::E5M2 => E5M2.get_or_init(|| build(0x80, 0x80)),
        FloatFormat::E2M1 => E2M1.get_or_init(|| build(8, 0x8)),
        other => unreachable!("{other:?} has no code grid"),
    }
}

/// The code nearest to `v` (saturating at the format's largest finite value; ties go to the even code,
/// the usual round-to-nearest-even of the format's mantissa). NaN encodes as zero.
fn nearest(g: &Grid, v: f32) -> u8 {
    let a = v.abs();
    if a.is_nan() {
        return 0;
    }
    let hi = g.mags.partition_point(|m| m.0 < a);
    let code = if hi == 0 {
        g.mags[0].1
    } else if hi == g.mags.len() {
        g.mags[hi - 1].1
    } else {
        let (lo, up) = (g.mags[hi - 1], g.mags[hi]);
        let (dl, du) = (a - lo.0, up.0 - a);
        if dl < du || (dl == du && lo.1 % 2 == 0) { lo.1 } else { up.1 }
    };
    if v.is_sign_negative() && code != 0 { code | g.sign } else { code }
}

fn max_finite(fmt: FloatFormat) -> f32 {
    grid(fmt).mags.last().map_or(1.0, |m| m.0)
}

/// Round `s` to the precision of a stored scale format.
fn as_scale(fmt: FloatFormat, s: f32) -> f32 {
    match fmt {
        FloatFormat::Bf16 => half::bf16::from_f32(s).to_f32(),
        _ => s,
    }
}

fn scale_bytes(fmt: FloatFormat, s: f32, out: &mut Vec<u8>) {
    match fmt {
        FloatFormat::Bf16 => out.extend_from_slice(&half::bf16::from_f32(s).to_le_bytes()),
        _ => out.extend_from_slice(&s.to_le_bytes()),
    }
}

fn absmax(xs: &[f32]) -> f32 {
    xs.iter().fold(0f32, |m, &v| if v.is_nan() { m } else { m.max(v.abs()) })
}

/// The geometry of a tiled-scale FP8 encoding: row length, code format, scale tile and scale format.
#[derive(Clone, Copy)]
struct Fp8Tiling {
    inner: usize,
    fmt: FloatFormat,
    rows: usize,
    cols: usize,
    scale_fmt: FloatFormat,
}

/// FP8 with one scale per `rows x cols` tile: scale = tile absmax / largest finite code (floored to 1
/// for an all-zero tile), stored at the scale format's precision and divided out with that stored value.
///
/// Row bands are independent (a band owns its payload rows and its run of scales), so a large tile is encoded by
/// contiguous groups of bands on scoped threads; the bytes are those of the serial loop.
fn encode_fp8_tiled(values: &[f32], inner: usize, fmt: FloatFormat, rows: usize, cols: usize, scale_fmt: FloatFormat, out: &mut TileBuf) {
    const PARALLEL_MIN_ELEMS: usize = 1 << 18;
    const MAX_THREADS: usize = 16;
    let tiling = Fp8Tiling { inner, fmt, rows, cols, scale_fmt };
    let n_rows = values.len() / inner;
    let bands = n_rows.div_ceil(rows);
    out.payload.clear();
    out.payload.resize(values.len(), 0);
    out.scales.push(Vec::new());
    let threads = if values.len() >= PARALLEL_MIN_ELEMS { std::thread::available_parallelism().map_or(1, |p| p.get()).min(MAX_THREADS).min(bands) } else { 1 };
    let per_thread = bands.div_ceil(threads.max(1));
    let mut parts: Vec<Vec<u8>> = vec![Vec::new(); bands.div_ceil(per_thread.max(1))];
    std::thread::scope(|scope| {
        let mut payload = out.payload.as_mut_slice();
        for (t, part) in parts.iter_mut().enumerate() {
            let (b0, b1) = (t * per_thread, ((t + 1) * per_thread).min(bands));
            let (r0, r1) = (b0 * rows, (b1 * rows).min(n_rows));
            let (mine, rest) = std::mem::take(&mut payload).split_at_mut((r1 - r0) * inner);
            payload = rest;
            scope.spawn(move || encode_fp8_bands(values, &tiling, (r0, r1), mine, part));
        }
    });
    for part in parts {
        out.scales[0].extend_from_slice(&part);
    }
}

/// The bands covering rows `r0..r1` (a multiple of `rows` from the tile's start): their payload into `payload` (which
/// starts at row `r0`) and their scales appended to `scales`.
fn encode_fp8_bands(values: &[f32], t: &Fp8Tiling, (r0, r1): (usize, usize), payload: &mut [u8], scales: &mut Vec<u8>) {
    let Fp8Tiling { inner, fmt, rows, cols, scale_fmt } = *t;
    let (g, top) = (grid(fmt), max_finite(fmt));
    let per_band = inner.div_ceil(cols);
    for band_r0 in (r0..r1).step_by(rows) {
        let band_r1 = (band_r0 + rows).min(r1);
        for cb in 0..per_band {
            let (c0, c1) = (cb * cols, ((cb + 1) * cols).min(inner));
            let amax = (band_r0..band_r1).flat_map(|r| values[r * inner + c0..r * inner + c1].iter()).fold(0f32, |m, &v| if v.is_nan() { m } else { m.max(v.abs()) });
            let s = if amax > 0.0 { as_scale(scale_fmt, amax / top) } else { 1.0 };
            let s = if s > 0.0 && s.is_finite() { s } else { 1.0 };
            scale_bytes(scale_fmt, s, scales);
            for r in band_r0..band_r1 {
                for c in c0..c1 {
                    payload[(r - r0) * inner + c] = nearest(g, values[r * inner + c] / s);
                }
            }
        }
    }
}

fn pack_nibbles(codes: impl Iterator<Item = u8>, n: usize, out: &mut Vec<u8>) {
    out.clear();
    out.resize(n.div_ceil(2), 0);
    for (i, c) in codes.enumerate() {
        out[i / 2] |= (c & 0xF) << (4 * (i % 2));
    }
}

/// NVFP4: the tensor scale is `tensor_absmax / (6 * 448)`, each 16-block scale is the E4M3 code nearest
/// to `block_absmax / (6 * tensor_scale)`, and each value the E2M1 code nearest to `v / (block scale *
/// tensor scale)` with the block scale as stored.
fn encode_nvfp4(values: &[f32], tensor_absmax: f32, out: &mut TileBuf) {
    let (g4, g2) = (grid(FloatFormat::E4M3Fn), grid(FloatFormat::E2M1));
    let (top2, top4) = (max_finite(FloatFormat::E2M1), max_finite(FloatFormat::E4M3Fn));
    let tensor_scale = if tensor_absmax > 0.0 { tensor_absmax / (top2 * top4) } else { 1.0 };
    out.scales.clear();
    out.scales.push(tensor_scale.to_le_bytes().to_vec());
    out.scales.push(Vec::with_capacity(values.len() / 16));
    let mut codes = Vec::with_capacity(values.len());
    for blk in values.chunks(16) {
        let sc = nearest(g4, absmax(blk) / (top2 * tensor_scale));
        out.scales[1].push(sc);
        let eff = super::decode_float_code(FloatFormat::E4M3Fn, sc as u32) * tensor_scale;
        codes.extend(blk.iter().map(|&v| if eff > 0.0 { nearest(g2, v / eff) } else { 0 }));
    }
    pack_nibbles(codes.into_iter(), values.len(), &mut out.payload);
}

/// MXFP4: the block scale is the power of two `2^(floor(log2(block_absmax)) - 2)` (E2M1's largest
/// exponent is 2, so the block maximum lands in [4, 8) and saturates at 6), stored as an E8M0 byte.
fn encode_mxfp4(values: &[f32], out: &mut TileBuf) {
    let g2 = grid(FloatFormat::E2M1);
    out.scales.clear();
    out.scales.push(Vec::with_capacity(values.len() / 32));
    let mut codes = Vec::with_capacity(values.len());
    for blk in values.chunks(32) {
        let amax = absmax(blk);
        let exp = if amax > 0.0 { (amax.log2().floor() as i32 - 2).clamp(-127, 127) } else { 0 };
        out.scales[0].push((exp + 127) as u8);
        let s = 2f32.powi(exp);
        codes.extend(blk.iter().map(|&v| nearest(g2, v / s)));
    }
    pack_nibbles(codes.into_iter(), values.len(), &mut out.payload);
}

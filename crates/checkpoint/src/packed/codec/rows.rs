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
}

pub(super) fn encode_kind(enc: &QuantEncoding) -> Option<EncodeKind> {
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
    out.clear();
    out.extend(vals.iter().flat_map(|v| v.to_le_bytes()));
}

/// Encode tile `elems` (`values`, row-major) of `desc` into `out`.
pub(super) fn encode(desc: &TensorDescriptor, elems: &Range<u64>, values: &[f32], out: &mut TileBuf) -> Result<(), String> {
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
    out.scales.clear();
    match kind {
        EncodeKind::Float(FloatFormat::F32) => f32s_to_le(values, &mut out.payload),
        EncodeKind::Float(FloatFormat::F16) => {
            out.payload.clear();
            out.payload.extend(values.iter().flat_map(|&v| half::f16::from_f32(v).to_le_bytes()));
        }
        EncodeKind::Float(_) => {
            out.payload.clear();
            out.payload.extend(values.iter().flat_map(|&v| half::bf16::from_f32(v).to_le_bytes()));
        }
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

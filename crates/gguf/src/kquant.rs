// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Host-side lossless relayout of GGUF K-quant/legacy block tensors into
//! brain's canonical device K-quant layout: ONE shape, three instantiations.
//!
//! Swedish Embedded AB implements quantized checkpoint import and device
//! layout tooling for its clients. If your team needs expertise in loading
//! GGUF K-quant checkpoints without a fp32 detour then you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! The canonical layout: `wq: [n, k*bits/32] u32` holds codes, K-contiguous,
//! `32/bits` codes packed per word (low bits first: code `b` of word `w`
//! occupies bits `[bits*b, bits*b+bits)` and covers element `w*(32/bits)+b`).
//! `wsz: [n, 2*k/G] f32` holds interleaved `(scale, min)` pairs, one pair per
//! `G`-element group of the reduction axis (`min` is always `0.0` for a
//! symmetric type). Six GGUF block formats land in this one shape:
//!
//! | type  | bits | G  | affine | codes                    |
//! |-------|------|----|--------|--------------------------|
//! | Q4_K  | 4    | 32 | yes    | raw nibble, `0..16`      |
//! | Q5_K  | 8    | 32 | yes    | raw 5-bit value, `0..32` |
//! | Q6_K  | 8    | 16 | no     | `q - 32`, signed         |
//! | Q5_0  | 8    | 32 | no     | `q - 16`, signed         |
//! | Q4_0  | 4    | 32 | no     | `q - 8`, signed          |
//! | Q8_0  | 8    | 32 | no     | `q`, signed              |
//!
//! Affine codes stay the format's own RAW unsigned value - never bias-folded
//! the way the symmetric family's codes are - because folding needs the
//! per-element `dm` term subtracted first, which is not an integer operation.
//! Symmetric codes are stored as their bias-folded SIGNED value in low-bits
//! two's complement, so [`pack_codes`]/[`unpack_row_codes`] need no per-type
//! special case: masking (pack) and optional sign-extension (unpack, gated on
//! `affine`) is the whole story.
//!
//! This module performs NO arithmetic on weight values. It re-derives exactly
//! the same `(scale, min)` products [`checkpoint::gguf`]'s private
//! `deq_q4_k`/`deq_q5_k`/`deq_q6_k`/`deq_q4_0`/`deq_q5_0`/`deq_q8_0` compute -
//! same expressions, same operand order, so `ds*code - dm` (affine) or
//! `ds*code` (symmetric) reproduces that oracle's output bit for bit - and
//! otherwise only moves codes between representations. `tests/kquant.rs`
//! gates that claim with `assert_eq!`, not a tolerance.

use checkpoint::gguf::GgmlType;
use checkpoint::TensorSource;

/// The device-layout parameters one relayout call produced, carried alongside
/// `(wq, wsz)` so a later GPU-upload milestone knows bits/group/affine
/// without re-deriving them from `ty`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KqLayout {
    /// The GGUF block format this came from.
    pub ty: GgmlType,
    /// Bits per code in `wq` (4 or 8 - not necessarily the format's own bit
    /// width; Q5_K's 5-bit code is stored in an 8-bit slot).
    pub bits: u32,
    /// Elements per weight-scale group along `k` (32, except Q6_K's 16).
    pub group: usize,
    /// Whether reconstruction needs `ds*code - dm` (true) or just `ds*code`
    /// (false, `dm` is always `0.0`).
    pub affine: bool,
    /// Output rows relaid out.
    pub n: usize,
    /// Reduction-axis length relaid out (elements, not words).
    pub k: usize,
}

impl KqLayout {
    /// `wq` words per output row: `k*bits/32`.
    pub fn words_per_row(&self) -> usize {
        self.k * self.bits as usize / 32
    }
    /// `wsz` groups per output row: `k/group`. `wsz` itself holds `2*` this
    /// many f32s per row (interleaved scale, min).
    pub fn groups_per_row(&self) -> usize {
        self.k / self.group
    }
}

/// `(bits, group, affine, block_elems)` for the six types this module
/// relays out. `None` for anything else - an IQ/TQ/MXFP4 codebook, Q2_K/
/// Q3_K/Q8_K (no device K-quant kernel targets them), or a plain F32/F16/
/// BF16 leaf.
fn geometry(ty: GgmlType) -> Option<(u32, usize, bool, usize)> {
    Some(match ty {
        GgmlType::Q4_0 => (4, 32, false, 32),
        GgmlType::Q5_0 => (8, 32, false, 32),
        GgmlType::Q8_0 => (8, 32, false, 32),
        GgmlType::Q4K => (4, 32, true, 256),
        GgmlType::Q5K => (8, 32, true, 256),
        GgmlType::Q6K => (8, 16, false, 256),
        _ => return None,
    })
}

/// Relayout the rectangle rows `[r0, r0+n_out)` x cols `[c0, c0+k)` of a
/// tensor stored as `[_, stride]` from its raw GGUF blocks into
/// `(wq, wsz, layout)` - this module's canonical device K-quant shape.
///
/// `None` means "fall back to the fp32 route", never a partial or
/// approximate answer: `source` has no raw blocks for `name` (see
/// [`checkpoint::TensorSource::raw_blocks`]), `name`'s type is not one of
/// the six this module handles, or `stride`/`c0`/`k` are not each a multiple
/// of the type's block size (32 for Q4_0/Q5_0/Q8_0, 256 for Q4_K/Q5_K/Q6_K -
/// a K-quant super-block cannot be split, so an unaligned `k` is a refusal
/// even when it would be a valid legacy-block boundary).
pub fn try_kq_rect(source: &dyn TensorSource, name: &str, stride: usize, r0: usize, n_out: usize, c0: usize, k: usize) -> Option<(Vec<u32>, Vec<f32>, KqLayout)> {
    let (block, raw) = source.raw_blocks(name)?;
    let ty = block.ty;
    let (bits, group, affine, block_elems) = geometry(ty)?;
    if !stride.is_multiple_of(block_elems) || !c0.is_multiple_of(block_elems) || !k.is_multiple_of(block_elems) {
        return None;
    }
    if n_out == 0 || k == 0 {
        return None;
    }
    let bb = ty.block_bytes();
    let nblocks_row = k / block_elems;
    // Every row starts on a block boundary because `stride` and `c0` are each
    // a multiple of `block_elems` (checked above), so `(r0+i)*stride+c0` is
    // too - no remainder to worry about when dividing by `block_elems`.
    let last_e0 = (r0 + n_out - 1) * stride + c0;
    let need_bytes = (last_e0 / block_elems + nblocks_row) * bb;
    if need_bytes > raw.len() {
        return None;
    }

    let gs = k / group;
    let words_per_row = k * bits as usize / 32;
    let mut wq = vec![0u32; n_out * words_per_row];
    let mut wsz = vec![0f32; n_out * 2 * gs];

    let mut codes: Vec<i32> = Vec::with_capacity(k);
    let mut ds: Vec<f32> = Vec::with_capacity(gs);
    let mut dm: Vec<f32> = Vec::with_capacity(gs);
    for i in 0..n_out {
        codes.clear();
        ds.clear();
        dm.clear();
        let e0 = (r0 + i) * stride + c0;
        let b0 = e0 / block_elems;
        for bi in 0..nblocks_row {
            let blk = &raw[(b0 + bi) * bb..(b0 + bi + 1) * bb];
            match ty {
                GgmlType::Q4_0 => relayout_q4_0(blk, &mut codes, &mut ds),
                GgmlType::Q5_0 => relayout_q5_0(blk, &mut codes, &mut ds),
                GgmlType::Q8_0 => relayout_q8_0(blk, &mut codes, &mut ds),
                GgmlType::Q4K => relayout_q4_k(blk, &mut codes, &mut ds, &mut dm),
                GgmlType::Q5K => relayout_q5_k(blk, &mut codes, &mut ds, &mut dm),
                GgmlType::Q6K => relayout_q6_k(blk, &mut codes, &mut ds),
                _ => unreachable!("geometry() only returns Some for these six types"),
            }
        }
        pack_codes(&codes, bits, &mut wq[i * words_per_row..(i + 1) * words_per_row]);
        for g in 0..gs {
            wsz[i * 2 * gs + 2 * g] = ds[g];
            wsz[i * 2 * gs + 2 * g + 1] = if affine { dm[g] } else { 0.0 };
        }
    }
    Some((wq, wsz, KqLayout { ty, bits, group, affine, n: n_out, k }))
}

/// Pack `codes` (one signed value per element, K-contiguous) into `out`
/// (`codes.len()*bits/32` words already sized by the caller): `32/bits`
/// codes per word, low bits first, each code masked to its low `bits` bits -
/// a two's complement truncation for a symmetric (already bias-folded)
/// code, a plain value for an affine (always non-negative, always `<
/// 2^bits`) one. The masking makes this the same operation either way.
fn pack_codes(codes: &[i32], bits: u32, out: &mut [u32]) {
    let per_word = 32 / bits as usize;
    let mask = (1u32 << bits) - 1;
    for (w, word_out) in out.iter_mut().enumerate() {
        let mut word = 0u32;
        for b in 0..per_word {
            let c = codes[w * per_word + b];
            word |= (c as u32 & mask) << (bits * b as u32);
        }
        *word_out = word;
    }
}

/// The inverse of [`pack_codes`]: unpack one row's `words` back to `layout.k`
/// signed code values. `layout.affine` selects whether a code's top bit is a
/// sign bit (symmetric, two's complement) or just part of the magnitude
/// (affine, always non-negative). Exposed (not just used by this module's own
/// tests) because a later GPU-upload milestone needs the identical mapping to
/// validate an uploaded buffer against the host source.
///
/// `words` must be exactly `layout.words_per_row()` long - `32/bits` codes
/// per word divides `layout.k` exactly by construction (`k` is a multiple of
/// the type's block size, which is itself a multiple of `32/bits`), so every
/// unpacked code is a real one, never packing padding.
pub fn unpack_row_codes(words: &[u32], layout: &KqLayout) -> Vec<i32> {
    let per_word = 32 / layout.bits as usize;
    let mask = (1u32 << layout.bits) - 1;
    let half = 1u32 << (layout.bits - 1);
    let mut out = Vec::with_capacity(layout.k);
    for &word in words {
        for b in 0..per_word {
            let raw = (word >> (layout.bits * b as u32)) & mask;
            let code = if layout.affine {
                raw as i32
            } else if raw >= half {
                raw as i32 - (1i32 << layout.bits)
            } else {
                raw as i32
            };
            out.push(code);
        }
    }
    out
}

/// f16 (ggml_half) at byte offset `i` in `b` - deliberately duplicated from
/// `checkpoint::gguf`'s own private helper (same `half` crate call) rather
/// than exposed across the crate boundary, since it is one line and the
/// bit-exactness this module gates depends on using the identical decode,
/// not on sharing code for its own sake.
fn f16(b: &[u8], i: usize) -> f32 {
    half::f16::from_le_bytes([b[i], b[i + 1]]).to_f32()
}

/// 6-bit packed scale/min extraction shared by Q4_K/Q5_K - duplicated from
/// `checkpoint::gguf`'s private `scale_min_k4` (ggml `get_scale_min_k4`) for
/// the same reason [`f16`] is: the bit-exactness gate needs the identical
/// expression, not a cross-crate reuse of it.
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

// ---- legacy blocks of 32 (one group == one block, min == 0) ----

fn relayout_q4_0(b: &[u8], codes: &mut Vec<i32>, ds: &mut Vec<f32>) {
    let d = f16(b, 0);
    let qs = &b[2..18];
    ds.push(d);
    let mut cb = [0i32; 32];
    for j in 0..16 {
        cb[j] = (qs[j] & 0x0F) as i32 - 8;
        cb[j + 16] = (qs[j] >> 4) as i32 - 8;
    }
    codes.extend_from_slice(&cb);
}

fn relayout_q5_0(b: &[u8], codes: &mut Vec<i32>, ds: &mut Vec<f32>) {
    let d = f16(b, 0);
    let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
    let qs = &b[6..22];
    ds.push(d);
    let mut cb = [0i32; 32];
    for j in 0..16 {
        let xh0 = (((qh >> j) << 4) & 0x10) as u8;
        let xh1 = ((qh >> (j + 12)) & 0x10) as u8;
        cb[j] = (((qs[j] & 0x0F) | xh0) as i32) - 16;
        cb[j + 16] = (((qs[j] >> 4) | xh1) as i32) - 16;
    }
    codes.extend_from_slice(&cb);
}

fn relayout_q8_0(b: &[u8], codes: &mut Vec<i32>, ds: &mut Vec<f32>) {
    let d = f16(b, 0);
    ds.push(d);
    for j in 0..32 {
        codes.push(b[2 + j] as i8 as i32);
    }
}

// ---- k-quant super-blocks of 256 ----

fn relayout_q4_k(b: &[u8], codes: &mut Vec<i32>, ds: &mut Vec<f32>, dm: &mut Vec<f32>) {
    let d = f16(b, 0);
    let dmin = f16(b, 2);
    let scales = &b[4..16];
    let qs = &b[16..144];
    // 8 groups of 32; group `g`'s nibble half and source byte range follow
    // `deq_q4_k`'s own `(qoff, is)` walk (outer step of 64 elements = one
    // lo/hi nibble pair): `qoff = 32*(g/2)`, low nibble for even `g`.
    for g in 0..8 {
        let (sc, m) = scale_min_k4(g, scales);
        ds.push(d * sc as f32);
        dm.push(dmin * m as f32);
        let qoff = 32 * (g / 2);
        let lo = g % 2 == 0;
        for l in 0..32 {
            let byte = qs[qoff + l];
            codes.push((if lo { byte & 0x0F } else { byte >> 4 }) as i32);
        }
    }
}

fn relayout_q5_k(b: &[u8], codes: &mut Vec<i32>, ds: &mut Vec<f32>, dm: &mut Vec<f32>) {
    let d = f16(b, 0);
    let dmin = f16(b, 2);
    let scales = &b[4..16];
    let qh = &b[16..48];
    let ql = &b[48..176];
    // Same 8-group walk as Q4_K, plus `deq_q5_k`'s high bit: group `g`'s
    // `qh` mask bit is `1 << (2*(g/2) + g%2)`, matching its `u1`/`u2` walk
    // (`u1 = 1<<2*outer`, `u2 = 2<<2*outer`) with no shift-tracking state.
    for g in 0..8 {
        let (sc, m) = scale_min_k4(g, scales);
        ds.push(d * sc as f32);
        dm.push(dmin * m as f32);
        let outer = g / 2;
        let qoff = 32 * outer;
        let shift = 2 * outer + (g % 2);
        let mask = 1u8 << shift;
        let lo = g % 2 == 0;
        for l in 0..32 {
            let base = if lo { ql[qoff + l] & 0x0F } else { ql[qoff + l] >> 4 };
            let hi = if qh[l] & mask != 0 { 16 } else { 0 };
            codes.push(base as i32 + hi);
        }
    }
}

fn relayout_q6_k(b: &[u8], codes: &mut Vec<i32>, ds: &mut Vec<f32>) {
    let ql = &b[0..128];
    let qh = &b[128..192];
    let sc = &b[192..208];
    let d = f16(b, 208);
    // `deq_q6_k`'s `sc` index (`sco+is` etc.) reduces to exactly the linear
    // 16-group index `pos/16` across the whole super-block once both `ni`
    // halves are laid flat - verified by construction, not asserted here:
    // `ni=0` covers `sc[0..8]` over output `[0,128)`, `ni=1` covers
    // `sc[8..16]` over `[128,256)`, and within each half the four
    // `(q1,q2,q3,q4)` sub-ranges of 32 land at consecutive 16-groups in
    // exactly `is`'s 0/1 order.
    let mut cb = [0i32; 256];
    for ni in 0..2 {
        let qlo = ni * 64;
        let qho = ni * 32;
        let yo = ni * 128;
        for l in 0..32 {
            let q1 = ((ql[qlo + l] & 0x0F) as i32 | ((qh[qho + l] & 3) as i32) << 4) - 32;
            let q2 = ((ql[qlo + l + 32] & 0x0F) as i32 | (((qh[qho + l] >> 2) & 3) as i32) << 4) - 32;
            let q3 = ((ql[qlo + l] >> 4) as i32 | (((qh[qho + l] >> 4) & 3) as i32) << 4) - 32;
            let q4 = ((ql[qlo + l + 32] >> 4) as i32 | (((qh[qho + l] >> 6) & 3) as i32) << 4) - 32;
            cb[yo + l] = q1;
            cb[yo + l + 32] = q2;
            cb[yo + l + 64] = q3;
            cb[yo + l + 96] = q4;
        }
    }
    codes.extend_from_slice(&cb);
    for &s in sc {
        ds.push(d * s as i8 as f32);
    }
}

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
//!
//! (M14) The weight-scale plane is no longer one flat `[n, 2*k/G] f32`
//! interleaved `(scale, min)` array (M8's original `wsz`). Every one of the
//! six formats this module targets already carries its scale as a PRODUCT of
//! two pieces at different granularity - a per-`G`-element-group sub-scale
//! (`sc`/`m`, `0..63` for the three K-quant formats, always `1`/`0` for the
//! three legacy formats which have no sub-scale at all) times a coarser f16
//! value shared by every group of one GGUF "super-block" (`d`/`dmin`, shared
//! by 8 groups for Q4_K/Q5_K, 16 for Q6_K, and trivially by exactly 1 group
//! for the three legacy formats, which have no super-block coarser than
//! their own 32-element block). Storing the ALREADY-MULTIPLIED f32 product
//! (M8's `wsz`) throws that structure away and pays a full f32 (4 bytes) per
//! group for information that is really a 1-byte sub-scale plus a 2-byte
//! value shared across many groups. M14 keeps the two pieces separate on the
//! device instead:
//!
//! * `wsm: [n, groups_per_row.div_ceil(2)] u32` - the per-group sub-scale
//!   pair, PACKED: group `g`'s `(sc, m)` u8 pair occupies bits `[0,16)` of
//!   word `g/2` when `g` is even, bits `[16,32)` when `g` is odd - i.e. two
//!   groups share one word, and within a group's own 16-bit half, `sc` is
//!   the LOW byte and `m` the HIGH byte. A row with an odd group count
//!   leaves the trailing half-word's upper bits unused (zero).
//! * `wd: [n, k/spb] u32` - the per-SUPER-BLOCK `(d, dmin)` f16 pair, PACKED:
//!   super-block `s`'s raw f16 BIT PATTERN for `d` occupies bits `[0,16)` of
//!   word `s`, `dmin`'s bits occupy `[16,32)`. `spb` ([`KqLayout::spb`]) is
//!   the format's own natural super-block element count: 256 for the three
//!   K-quant formats (their real GGUF super-block), 32 for the three legacy
//!   formats (== their own block size - they have no coarser grouping, so
//!   `wd` there has exactly one entry per group and [`wsm`]'s sub-scale byte
//!   is always the trivial multiplicative identity `sc=1, m=0`).
//!
//! Reconstruction is `ds = f16_from_bits(wd_d_half).to_f32() * f32(sc)`,
//! `dm = f16_from_bits(wd_dmin_half).to_f32() * f32(m)` - and because `wd`
//! stores the format's OWN raw f16 bit pattern (not a value round-tripped
//! through any arithmetic) and `sc`/`m` are the format's own `0..63` (or
//! trivial `1`/`0`) integers losslessly widened to `u8`, this reproduces
//! `checkpoint::gguf`'s `d*sc`/`dmin*m` oracle expressions bit for bit - the
//! same claim M8's `wsz` made, restated for the packed encoding.
//!
//! Q6_K's own per-group sub-scale is a SIGNED `i8` (unlike Q4_K/Q5_K's
//! unsigned 6-bit value) - [`KqLayout::scale_signed`] records this so a
//! decoder knows to sign-extend the stored byte before use; every OTHER
//! format's sub-scale is a plain unsigned value (the three legacy formats'
//! constant `1` doesn't care either way).
//!
//! Six GGUF block formats land in this one shape:
//!
//! | type  | bits | G  | affine | spb | codes                    |
//! |-------|------|----|--------|-----|--------------------------|
//! | Q4_K  | 4    | 32 | yes    | 256 | raw nibble, `0..16`      |
//! | Q5_K  | 8    | 32 | yes    | 256 | raw 5-bit value, `0..32` |
//! | Q6_K  | 8    | 16 | no     | 256 | `q - 32`, signed         |
//! | Q5_0  | 8    | 32 | no     | 32  | `q - 16`, signed         |
//! | Q4_0  | 4    | 32 | no     | 32  | `q - 8`, signed          |
//! | Q8_0  | 8    | 32 | no     | 32  | `q`, signed              |
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
//! otherwise only moves codes/scales between representations. `tests/
//! kquant.rs` gates that claim with `assert_eq!`, not a tolerance.

use checkpoint::gguf::GgmlType;
use checkpoint::TensorSource;

/// The device-layout parameters one relayout call produced, carried alongside
/// `(wq, wsm, wd)` so a later GPU-upload milestone knows bits/group/affine/
/// spb without re-deriving them from `ty`.
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
    /// Elements sharing one `wd` `(d, dmin)` f16 pair - 256 for the three
    /// K-quant super-block formats, 32 (== `group`'s own block size) for the
    /// three legacy formats, which have no coarser grouping at all.
    pub spb: usize,
    /// Whether the per-group sub-scale byte stored in `wsm` is SIGNED (only
    /// Q6_K - `i8` in `-128..127`) or a plain unsigned value (every other
    /// format, including the three legacy formats' trivial constant `1`).
    pub scale_signed: bool,
}

impl KqLayout {
    /// `wq` words per output row: `k*bits/32`.
    pub fn words_per_row(&self) -> usize {
        self.k * self.bits as usize / 32
    }
    /// Weight-scale groups per output row: `k/group`.
    pub fn groups_per_row(&self) -> usize {
        self.k / self.group
    }
    /// `wsm` words per output row: two groups' `(sc, m)` pairs per word.
    pub fn wsm_words_per_row(&self) -> usize {
        self.groups_per_row().div_ceil(2)
    }
    /// `wd` words per output row: one `(d, dmin)` pair per super-block.
    pub fn wd_words_per_row(&self) -> usize {
        self.k / self.spb
    }
    /// Weight-scale groups sharing one `wd` super-block entry: `spb/group`
    /// (8 for Q4_K/Q5_K, 16 for Q6_K, 1 for every legacy format).
    pub fn groups_per_superblock(&self) -> usize {
        self.spb / self.group
    }
}

/// `(bits, group, affine, block_elems)` for the six types this module
/// relays out. `None` for anything else - an IQ/TQ/MXFP4 codebook, Q2_K/
/// Q3_K/Q8_K (no device K-quant kernel targets them), or a plain F32/F16/
/// BF16 leaf. `block_elems` doubles as [`KqLayout::spb`] - every one of
/// these six formats' own natural super-block (the GGUF K-quant formats'
/// real 256-element super-block, or a legacy format's own 32-element block,
/// which has no coarser grouping) is exactly this module's block-alignment
/// unit already, so no separate "spb" table is needed.
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
/// tensor stored as `[_, stride]` from its raw GGUF blocks into `(wq, wsm,
/// wd, layout)` - this module's canonical device K-quant shape.
///
/// `None` means "fall back to the fp32 route", never a partial or
/// approximate answer: `source` has no raw blocks for `name` (see
/// [`checkpoint::TensorSource::raw_blocks`]), `name`'s type is not one of
/// the six this module handles, or `stride`/`c0`/`k` are not each a multiple
/// of the type's block size (32 for Q4_0/Q5_0/Q8_0, 256 for Q4_K/Q5_K/Q6_K -
/// a K-quant super-block cannot be split, so an unaligned `k` is a refusal
/// even when it would be a valid legacy-block boundary).
pub fn try_kq_rect(source: &dyn TensorSource, name: &str, stride: usize, r0: usize, n_out: usize, c0: usize, k: usize) -> Option<(Vec<u32>, Vec<u32>, Vec<u32>, KqLayout)> {
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

    let layout = KqLayout { ty, bits, group, affine, n: n_out, k, spb: block_elems, scale_signed: matches!(ty, GgmlType::Q6K) };
    let gs = layout.groups_per_row();
    let words_per_row = layout.words_per_row();
    let wsm_words_per_row = layout.wsm_words_per_row();
    let wd_words_per_row = layout.wd_words_per_row();
    let mut wq = vec![0u32; n_out * words_per_row];
    let mut wsm = vec![0u32; n_out * wsm_words_per_row];
    let mut wd = vec![0u32; n_out * wd_words_per_row];

    let mut codes: Vec<i32> = Vec::with_capacity(k);
    let mut d_bits: Vec<u16> = Vec::with_capacity(nblocks_row);
    let mut dmin_bits: Vec<u16> = Vec::with_capacity(nblocks_row);
    let mut sc: Vec<u8> = Vec::with_capacity(gs);
    let mut mn: Vec<u8> = Vec::with_capacity(gs);
    for i in 0..n_out {
        codes.clear();
        d_bits.clear();
        dmin_bits.clear();
        sc.clear();
        mn.clear();
        let e0 = (r0 + i) * stride + c0;
        let b0 = e0 / block_elems;
        for bi in 0..nblocks_row {
            let blk = &raw[(b0 + bi) * bb..(b0 + bi + 1) * bb];
            match ty {
                GgmlType::Q4_0 => relayout_q4_0(blk, &mut codes, &mut d_bits, &mut dmin_bits, &mut sc, &mut mn),
                GgmlType::Q5_0 => relayout_q5_0(blk, &mut codes, &mut d_bits, &mut dmin_bits, &mut sc, &mut mn),
                GgmlType::Q8_0 => relayout_q8_0(blk, &mut codes, &mut d_bits, &mut dmin_bits, &mut sc, &mut mn),
                GgmlType::Q4K => relayout_q4_k(blk, &mut codes, &mut d_bits, &mut dmin_bits, &mut sc, &mut mn),
                GgmlType::Q5K => relayout_q5_k(blk, &mut codes, &mut d_bits, &mut dmin_bits, &mut sc, &mut mn),
                GgmlType::Q6K => relayout_q6_k(blk, &mut codes, &mut d_bits, &mut dmin_bits, &mut sc, &mut mn),
                _ => unreachable!("geometry() only returns Some for these six types"),
            }
        }
        pack_codes(&codes, bits, &mut wq[i * words_per_row..(i + 1) * words_per_row]);
        let wsm_row = &mut wsm[i * wsm_words_per_row..(i + 1) * wsm_words_per_row];
        for g in 0..gs {
            let (w, shift) = (g / 2, if g % 2 == 0 { 0u32 } else { 16u32 });
            wsm_row[w] |= (sc[g] as u32) << shift;
            wsm_row[w] |= (mn[g] as u32) << (shift + 8);
        }
        let wd_row = &mut wd[i * wd_words_per_row..(i + 1) * wd_words_per_row];
        for s in 0..nblocks_row {
            wd_row[s] = (d_bits[s] as u32) | ((dmin_bits[s] as u32) << 16);
        }
    }
    Some((wq, wsm, wd, layout))
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

/// One weight-scale group's packed `(sc, m)` sub-scale byte pair out of one
/// row's `wsm` words (see this module's own doc comment for the exact bit
/// layout). `sc`/`m` are the raw stored bytes - a caller that needs Q6_K's
/// signed interpretation applies [`KqLayout::scale_signed`] itself, since
/// this function has no `KqLayout` to consult.
pub fn unpack_group_scale(wsm: &[u32], g: usize) -> (u8, u8) {
    let (w, shift) = (g / 2, if g % 2 == 0 { 0u32 } else { 16u32 });
    let word = wsm[w];
    (((word >> shift) & 0xFF) as u8, ((word >> (shift + 8)) & 0xFF) as u8)
}

/// One super-block's `(d, dmin)` f16 pair, already widened to f32 - EXACT,
/// since a f16 value widens to f32 losslessly (no rounding at all, unlike
/// the narrowing direction).
pub fn unpack_superblock_d(wd: &[u32], sblk: usize) -> (f32, f32) {
    let word = wd[sblk];
    (half::f16::from_bits((word & 0xFFFF) as u16).to_f32(), half::f16::from_bits((word >> 16) as u16).to_f32())
}

/// Reconstruct one row's full `(ds, dm)` per-group plane from `(wsm, wd)` -
/// `ds*code - dm` (affine) / `ds*code` (symmetric, `dm` is always `0.0` by
/// construction: `dmin_bits`/`m` are always `0` for a non-affine type) then
/// reproduces the oracle exactly, the same claim [`try_kq_rect`]'s own doc
/// comment makes for the packed encoding.
pub fn unpack_row_scales(wsm: &[u32], wd: &[u32], layout: &KqLayout) -> (Vec<f32>, Vec<f32>) {
    let gs = layout.groups_per_row();
    let gps = layout.groups_per_superblock();
    let mut ds = Vec::with_capacity(gs);
    let mut dm = Vec::with_capacity(gs);
    for g in 0..gs {
        let (d, dmin) = unpack_superblock_d(wd, g / gps);
        let (scb, mb) = unpack_group_scale(wsm, g);
        let scv = if layout.scale_signed { scb as i8 as f32 } else { scb as f32 };
        ds.push(d * scv);
        dm.push(dmin * mb as f32);
    }
    (ds, dm)
}

/// f16 (ggml_half) RAW BIT PATTERN at byte offset `i` in `b` - the little-
/// endian u16 GGUF stores, unconverted. Kept as bits (not widened to f32
/// here) so [`try_kq_rect`] can pack it straight into `wd` with no
/// round-trip: `half::f16::from_bits(raw).to_f32()` at decode time
/// reproduces exactly what the OLD `f16(b,i)` helper's `half::f16::
/// from_le_bytes(..).to_f32()` computed, since both parse the identical bit
/// pattern.
fn f16_bits(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}

/// 6-bit packed scale/min extraction shared by Q4_K/Q5_K - duplicated from
/// `checkpoint::gguf`'s private `scale_min_k4` (ggml `get_scale_min_k4`) for
/// the same reason [`f16_bits`] is: the bit-exactness gate needs the
/// identical expression, not a cross-crate reuse of it.
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

// ---- legacy blocks of 32 (one group == one block == one "super-block") ----
//
// None of the three legacy formats has any scale coarser than its own
// per-block f16 `d` - there is no separate sub-scale to extract, so `sc` is
// pushed as the trivial multiplicative identity `1` (`d * 1 == d`, exact,
// no rounding) and `m`/`dmin_bits` are always `0` (these are symmetric
// types, `dm` is always `0.0`).

fn relayout_q4_0(b: &[u8], codes: &mut Vec<i32>, d_bits: &mut Vec<u16>, dmin_bits: &mut Vec<u16>, sc: &mut Vec<u8>, mn: &mut Vec<u8>) {
    d_bits.push(f16_bits(b, 0));
    dmin_bits.push(0);
    sc.push(1);
    mn.push(0);
    let qs = &b[2..18];
    let mut cb = [0i32; 32];
    for j in 0..16 {
        cb[j] = (qs[j] & 0x0F) as i32 - 8;
        cb[j + 16] = (qs[j] >> 4) as i32 - 8;
    }
    codes.extend_from_slice(&cb);
}

fn relayout_q5_0(b: &[u8], codes: &mut Vec<i32>, d_bits: &mut Vec<u16>, dmin_bits: &mut Vec<u16>, sc: &mut Vec<u8>, mn: &mut Vec<u8>) {
    d_bits.push(f16_bits(b, 0));
    dmin_bits.push(0);
    sc.push(1);
    mn.push(0);
    let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
    let qs = &b[6..22];
    let mut cb = [0i32; 32];
    for j in 0..16 {
        let xh0 = (((qh >> j) << 4) & 0x10) as u8;
        let xh1 = ((qh >> (j + 12)) & 0x10) as u8;
        cb[j] = (((qs[j] & 0x0F) | xh0) as i32) - 16;
        cb[j + 16] = (((qs[j] >> 4) | xh1) as i32) - 16;
    }
    codes.extend_from_slice(&cb);
}

fn relayout_q8_0(b: &[u8], codes: &mut Vec<i32>, d_bits: &mut Vec<u16>, dmin_bits: &mut Vec<u16>, sc: &mut Vec<u8>, mn: &mut Vec<u8>) {
    d_bits.push(f16_bits(b, 0));
    dmin_bits.push(0);
    sc.push(1);
    mn.push(0);
    for j in 0..32 {
        codes.push(b[2 + j] as i8 as i32);
    }
}

// ---- k-quant super-blocks of 256 ----

fn relayout_q4_k(b: &[u8], codes: &mut Vec<i32>, d_bits: &mut Vec<u16>, dmin_bits: &mut Vec<u16>, sc: &mut Vec<u8>, mn: &mut Vec<u8>) {
    d_bits.push(f16_bits(b, 0));
    dmin_bits.push(f16_bits(b, 2));
    let scales = &b[4..16];
    let qs = &b[16..144];
    // 8 groups of 32; group `g`'s nibble half and source byte range follow
    // `deq_q4_k`'s own `(qoff, is)` walk (outer step of 64 elements = one
    // lo/hi nibble pair): `qoff = 32*(g/2)`, low nibble for even `g`.
    for g in 0..8 {
        let (s, m) = scale_min_k4(g, scales);
        sc.push(s);
        mn.push(m);
        let qoff = 32 * (g / 2);
        let lo = g % 2 == 0;
        for l in 0..32 {
            let byte = qs[qoff + l];
            codes.push((if lo { byte & 0x0F } else { byte >> 4 }) as i32);
        }
    }
}

fn relayout_q5_k(b: &[u8], codes: &mut Vec<i32>, d_bits: &mut Vec<u16>, dmin_bits: &mut Vec<u16>, sc: &mut Vec<u8>, mn: &mut Vec<u8>) {
    d_bits.push(f16_bits(b, 0));
    dmin_bits.push(f16_bits(b, 2));
    let scales = &b[4..16];
    let qh = &b[16..48];
    let ql = &b[48..176];
    // Same 8-group walk as Q4_K, plus `deq_q5_k`'s high bit: group `g`'s
    // `qh` mask bit is `1 << (2*(g/2) + g%2)`, matching its `u1`/`u2` walk
    // (`u1 = 1<<2*outer`, `u2 = 2<<2*outer`) with no shift-tracking state.
    for g in 0..8 {
        let (s, m) = scale_min_k4(g, scales);
        sc.push(s);
        mn.push(m);
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

fn relayout_q6_k(b: &[u8], codes: &mut Vec<i32>, d_bits: &mut Vec<u16>, dmin_bits: &mut Vec<u16>, sc: &mut Vec<u8>, mn: &mut Vec<u8>) {
    let ql = &b[0..128];
    let qh = &b[128..192];
    let scb = &b[192..208];
    d_bits.push(f16_bits(b, 208));
    // Symmetric - `dm` is always `0.0`, so `dmin_bits` is pushed as the
    // trivial `0` bit pattern (`f16(0.0).to_bits() == 0`) once per
    // super-block, matching every OTHER relayout function's convention (the
    // shared packing loop in `try_kq_rect` indexes `dmin_bits[s]`
    // unconditionally, regardless of `affine`).
    dmin_bits.push(0);
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
    // Q6_K's own sub-scale is a SIGNED i8 (unlike Q4_K/Q5_K's unsigned 6-bit
    // value) - stored as the raw byte pattern here; `KqLayout::scale_signed`
    // tells a decoder to `as i8` it back before use (see this module's own
    // doc comment).
    for &s in scb {
        sc.push(s);
        mn.push(0);
    }
}

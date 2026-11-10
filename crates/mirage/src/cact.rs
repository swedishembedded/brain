// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Reader for the `.cact` deployment container and its Cactus-Quant matrices.
//!
//! Layout (all little-endian): a 196-byte header (a tag, 47 more `u32`
//! geometry fields and a rotary base as `f32`), 28 shared codebook floats,
//! a nameless directory of 44-byte records, then 64-byte-aligned blobs.
//! Tensors are positional: what entry `i` means is decided by the model
//! configuration ([`crate::cactmap`]), not by the file.
//!
//! A Cactus-Quant ("CQ") matrix is `[out, in]` with `in` padded up to whole
//! groups (normally 128). Each group holds codebook indices packed LSB-first,
//! `bits` bits each, rows contiguous; after all index rows come the fp16 group
//! norms `[out, groups]`. A group decodes to `(codebook[idx] * norm)` rotated
//! by the natural-order Walsh matrix over the group, scaled `1/sqrt(group)`.
//!
//! The file is hostile input. Every offset and length is range-checked with
//! checked arithmetic before it is used, unknown dtypes and widths are
//! refused, and the reader owns the bytes, so no accessor can read outside
//! them.
//!
//! Swedish Embedded AB implements solutions for running aggressively quantised
//! models on constrained devices for its clients. If your team needs expertise
//! in low-bit weight formats then you can procure our services by sending an
//! email to info@swedishembedded.com.

use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};

/// The tag every `.cact` file starts with.
pub const TAG: u32 = 0x05E1_2A84;

/// `u32` fields in the header, the tag included.
pub const HEADER_U32S: usize = 48;

/// Header bytes: the `u32` fields and the trailing `f32` rotary base.
pub const HEADER_BYTES: usize = HEADER_U32S * 4 + 4;

/// Codebook floats: 2-bit (4 levels), 3-bit (8) and 4-bit (16) concatenated.
pub const CODEBOOK_LEN: usize = 28;

/// Bytes of one directory record.
pub const RECORD_BYTES: usize = 44;

/// Offset of the first directory record.
pub const DIRECTORY_START: usize = HEADER_BYTES + CODEBOOK_LEN * 4;

/// The most tensors a file may declare.
pub const MAX_TENSORS: usize = 1 << 16;

/// The largest file the reader will load.
pub const MAX_FILE_BYTES: u64 = 1 << 30;

/// Largest and smallest quantisation group.
const MAX_GROUP: usize = 4096;
const MIN_GROUP: usize = 8;

/// Why a `.cact` file was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CactError {
    /// The file could not be read.
    Io { path: PathBuf, message: String },
    /// The header, codebooks or directory table is malformed.
    Header(String),
    /// One directory entry is malformed, or was used as the wrong kind.
    Entry { index: usize, reason: String },
}

impl fmt::Display for CactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CactError::Io { path, message } => write!(f, "cact: {}: {message}", path.display()),
            CactError::Header(m) => write!(f, "cact: header: {m}"),
            CactError::Entry { index, reason } => write!(f, "cact: entry {index}: {reason}"),
        }
    }
}

impl std::error::Error for CactError {}

fn entry_error(index: usize, reason: impl Into<String>) -> CactError {
    CactError::Entry { index, reason: reason.into() }
}

/// The fixed-size header: the geometry the exporter recorded.
#[derive(Debug, Clone, PartialEq)]
pub struct CactHeader {
    /// All 48 `u32` fields, the tag first.
    pub fields: [u32; HEADER_U32S],
    /// The decoder's rotary base.
    pub rope_theta: f32,
}

impl CactHeader {
    /// Directory entries.
    pub fn num_tensors(&self) -> usize {
        self.fields[1] as usize
    }
    /// Embedding rows (the tied vocabulary).
    pub fn vocab(&self) -> usize {
        self.fields[5] as usize
    }
    /// Model width.
    pub fn d_model(&self) -> usize {
        self.fields[7] as usize
    }
    /// Query heads.
    pub fn num_heads(&self) -> usize {
        self.fields[8] as usize
    }
    /// Key/value heads.
    pub fn num_kv_heads(&self) -> usize {
        self.fields[9] as usize
    }
    /// Decoder layers.
    pub fn num_layers(&self) -> usize {
        self.fields[10] as usize
    }
    /// Query/key head width.
    pub fn qk_head_dim(&self) -> usize {
        self.fields[11] as usize
    }
    /// Value head width.
    pub fn v_head_dim(&self) -> usize {
        self.fields[12] as usize
    }
    /// Sequence limit.
    pub fn max_seq_len(&self) -> usize {
        self.fields[13] as usize
    }
    /// Width of the Hadamard MLP.
    pub fn hada_n(&self) -> usize {
        self.fields[14] as usize
    }
    /// Hyper-connection lanes.
    pub fn mhc_lanes(&self) -> usize {
        self.fields[15] as usize
    }
    /// Convolution taps on q, k and v.
    pub fn qkv_conv_taps(&self) -> usize {
        self.fields[19] as usize
    }
    /// Slots per engram table.
    pub fn engram_slots(&self) -> usize {
        self.fields[20] as usize
    }
    /// Width of one engram table row.
    pub fn engram_sub_dim(&self) -> usize {
        self.fields[21] as usize
    }
    /// Engram tables per site.
    pub fn num_engram_tables(&self) -> usize {
        self.fields[22] as usize
    }
    /// Engram value-convolution taps.
    pub fn engram_conv_taps(&self) -> usize {
        self.fields[23] as usize
    }
    /// N-gram orders of the engram hashes.
    pub fn engram_orders(&self) -> Vec<usize> {
        self.fields[27..27 + self.fields[26] as usize].iter().map(|&v| v as usize).collect()
    }
    /// Decoder layers that read an engram.
    pub fn engram_sites(&self) -> Vec<usize> {
        self.fields[32..32 + self.fields[31] as usize].iter().map(|&v| v as usize).collect()
    }
}

/// How an entry's bytes are to be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryDtype {
    /// IEEE half precision.
    Fp16,
    /// IEEE single precision.
    Fp32,
    /// Cactus-Quant matrix.
    Cq,
    /// Opaque bytes (the tokenizer).
    Raw,
}

/// One directory record, validated against the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// How to read the blob.
    pub dtype: EntryDtype,
    /// Logical shape (empty for a raw blob).
    pub shape: Vec<usize>,
    /// Blob offset in the file.
    pub offset: usize,
    /// Blob length in bytes.
    pub nbytes: usize,
    /// CQ group size (0 otherwise).
    pub group: usize,
    /// CQ index width in bits (0 otherwise).
    pub bits: u32,
}

impl Entry {
    /// Logical element count (1 for a raw blob's empty shape is not meaningful).
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3], b[at + 4], b[at + 5], b[at + 6], b[at + 7]])
}

/// A parsed, fully validated `.cact` file.
#[derive(Debug)]
pub struct CactFile {
    bytes: Vec<u8>,
    header: CactHeader,
    codebooks: [f32; CODEBOOK_LEN],
    entries: Vec<Entry>,
}

impl CactFile {
    /// Read and validate a file.
    pub fn open(path: &Path) -> Result<CactFile, CactError> {
        let io = |e: std::io::Error| CactError::Io { path: path.to_path_buf(), message: e.to_string() };
        let mut file = std::fs::File::open(path).map_err(io)?;
        let len = file.metadata().map_err(io)?.len();
        if len > MAX_FILE_BYTES {
            return Err(CactError::Header(format!("{len} bytes is too large for a model file (limit {MAX_FILE_BYTES})")));
        }
        let mut bytes = Vec::with_capacity(len as usize);
        // `take` bounds the read even if the file grows while it is open.
        (&mut file).take(MAX_FILE_BYTES).read_to_end(&mut bytes).map_err(io)?;
        CactFile::parse(bytes)
    }

    /// Validate `bytes` as a `.cact` file.
    pub fn parse(bytes: Vec<u8>) -> Result<CactFile, CactError> {
        if bytes.len() < DIRECTORY_START {
            return Err(CactError::Header(format!("{} bytes is shorter than the {DIRECTORY_START}-byte header and codebooks", bytes.len())));
        }
        let mut fields = [0u32; HEADER_U32S];
        for (i, f) in fields.iter_mut().enumerate() {
            *f = u32_at(&bytes, i * 4);
        }
        if fields[0] != TAG {
            return Err(CactError::Header(format!("tag {:#010x} is not {TAG:#010x}", fields[0])));
        }
        let rope_theta = f32::from_bits(u32_at(&bytes, HEADER_U32S * 4));
        let header = CactHeader { fields, rope_theta };
        if fields[2] as usize != CODEBOOK_LEN {
            return Err(CactError::Header(format!("codebook length {} is not {CODEBOOK_LEN} (2-, 3- and 4-bit books)", fields[2])));
        }
        if fields[26] as usize > 4 || fields[31] as usize > 16 {
            return Err(CactError::Header(format!("{} engram orders and {} sites exceed the header's 4 and 16 slots", fields[26], fields[31])));
        }
        let mut codebooks = [0f32; CODEBOOK_LEN];
        for (i, c) in codebooks.iter_mut().enumerate() {
            *c = f32::from_bits(u32_at(&bytes, HEADER_BYTES + i * 4));
            if !c.is_finite() {
                return Err(CactError::Header(format!("codebook value {i} is not finite")));
            }
        }
        let n = header.num_tensors();
        if n > MAX_TENSORS {
            return Err(CactError::Header(format!("{n} tensors exceeds the limit of {MAX_TENSORS}")));
        }
        let directory_end = n
            .checked_mul(RECORD_BYTES)
            .and_then(|d| d.checked_add(DIRECTORY_START))
            .filter(|&end| end <= bytes.len())
            .ok_or_else(|| CactError::Header(format!("the directory of {n} tensors does not fit in {} bytes", bytes.len())))?;
        let mut entries = Vec::with_capacity(n);
        for index in 0..n {
            entries.push(parse_entry(&bytes, index, DIRECTORY_START + index * RECORD_BYTES, directory_end)?);
        }
        Ok(CactFile { bytes, header, codebooks, entries })
    }

    /// The header.
    pub fn header(&self) -> &CactHeader {
        &self.header
    }

    /// The directory, in file order.
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// The shared codebook for `bits`-wide indices.
    pub fn codebook(&self, bits: u32) -> Option<&[f32]> {
        match bits {
            2 => Some(&self.codebooks[0..4]),
            3 => Some(&self.codebooks[4..12]),
            4 => Some(&self.codebooks[12..28]),
            _ => None,
        }
    }

    fn entry(&self, index: usize) -> Result<&Entry, CactError> {
        self.entries.get(index).ok_or_else(|| entry_error(index, format!("there are only {} entries", self.entries.len())))
    }

    fn blob(&self, e: &Entry) -> &[u8] {
        // Offsets and lengths were checked against the file at parse time.
        &self.bytes[e.offset..e.offset + e.nbytes]
    }

    /// The bytes of a raw entry.
    pub fn raw(&self, index: usize) -> Result<&[u8], CactError> {
        let e = self.entry(index)?;
        if e.dtype != EntryDtype::Raw {
            return Err(entry_error(index, format!("is {:?}, not a raw blob", e.dtype)));
        }
        Ok(self.blob(e))
    }

    /// The quantised matrix view of a CQ entry (packed indices and norms stay
    /// in place for native kernels).
    pub fn cq(&self, index: usize) -> Result<CqMatrix<'_>, CactError> {
        let e = self.entry(index)?;
        if e.dtype != EntryDtype::Cq {
            return Err(entry_error(index, format!("is {:?}, not a CQ matrix", e.dtype)));
        }
        let codebook = self.codebook(e.bits).ok_or_else(|| entry_error(index, format!("no codebook for {} bits", e.bits)))?;
        let rows = e.shape[0];
        let cols: usize = e.shape[1..].iter().product();
        let padded_cols = cols.div_ceil(e.group) * e.group;
        let packed_len = rows * padded_cols * e.bits as usize / 8;
        let blob = self.blob(e);
        Ok(CqMatrix { rows, cols, padded_cols, bits: e.bits, group: e.group, packed: &blob[..packed_len], norms: &blob[packed_len..], codebook })
    }

    /// Any numeric entry as row-major `f32`; a CQ entry is decoded.
    pub fn f32_tensor(&self, index: usize) -> Result<Vec<f32>, CactError> {
        let e = self.entry(index)?;
        match e.dtype {
            EntryDtype::Fp16 => Ok(self.blob(e).as_chunks::<2>().0.iter().map(|c| half::f16::from_le_bytes(*c).to_f32()).collect()),
            EntryDtype::Fp32 => Ok(self.blob(e).as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()),
            EntryDtype::Cq => Ok(self.cq(index)?.to_f32()),
            EntryDtype::Raw => Err(entry_error(index, "is a raw blob, not a numeric tensor")),
        }
    }
}

fn parse_entry(bytes: &[u8], index: usize, at: usize, directory_end: usize) -> Result<Entry, CactError> {
    let dtype = match bytes[at] {
        1 => EntryDtype::Fp16,
        2 => EntryDtype::Fp32,
        3 => EntryDtype::Cq,
        4 => EntryDtype::Raw,
        other => return Err(entry_error(index, format!("dtype {other} is not 1 (fp16), 2 (fp32), 3 (cq) or 4 (raw)"))),
    };
    let ndim = bytes[at + 1] as usize;
    if ndim > 4 {
        return Err(entry_error(index, format!("ndim {ndim} exceeds 4")));
    }
    let dims: Vec<u32> = (0..4).map(|d| u32_at(bytes, at + 4 + 4 * d)).collect();
    if dims[ndim..].iter().any(|&d| d != 0) {
        return Err(entry_error(index, format!("shape {dims:?} has non-zero extents beyond ndim {ndim}")));
    }
    let shape: Vec<usize> = dims[..ndim].iter().map(|&d| d as usize).collect();
    let offset = u64_at(bytes, at + 20);
    let nbytes = u64_at(bytes, at + 28);
    let group = u32_at(bytes, at + 36) as usize;
    let bits = u32_at(bytes, at + 40);

    let end = offset.checked_add(nbytes).filter(|&e| e <= bytes.len() as u64);
    let (Some(_), true) = (end, offset >= directory_end as u64) else {
        return Err(entry_error(
            index,
            if end.is_none() {
                format!("blob [{offset}, +{nbytes}) lies outside the {}-byte file", bytes.len())
            } else {
                format!("blob offset {offset} lies inside the header or directory (ends at {directory_end})")
            },
        ));
    };
    let (offset, nbytes) = (offset as usize, nbytes as usize);
    let numel = shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d)).ok_or_else(|| entry_error(index, format!("shape {shape:?} overflows")))?;

    let expect = |label: &str, expected: Option<usize>| match expected {
        Some(e) if e == nbytes => Ok(()),
        Some(e) => Err(entry_error(index, format!("{label} blob has {nbytes} bytes, its shape {shape:?} needs {e}"))),
        None => Err(entry_error(index, format!("{label} blob size for shape {shape:?} overflows"))),
    };
    match dtype {
        EntryDtype::Fp16 => expect("fp16", numel.checked_mul(2))?,
        EntryDtype::Fp32 => expect("fp32", numel.checked_mul(4))?,
        EntryDtype::Raw => {}
        EntryDtype::Cq => {
            if !(2..=4).contains(&bits) {
                return Err(entry_error(index, format!("{bits} bits has no codebook (supported: 2, 3, 4)")));
            }
            if !group.is_power_of_two() || !(MIN_GROUP..=MAX_GROUP).contains(&group) {
                return Err(entry_error(index, format!("group {group} is not a power of two in [{MIN_GROUP}, {MAX_GROUP}]")));
            }
            if ndim < 2 || shape.contains(&0) {
                return Err(entry_error(index, format!("shape {shape:?} is not a non-empty matrix")));
            }
            let rows = shape[0];
            let cols = shape[1..].iter().try_fold(1usize, |acc, &d| acc.checked_mul(d));
            let needed = cols.and_then(|c| {
                let padded = c.div_ceil(group).checked_mul(group)?;
                let packed = rows.checked_mul(padded)?.checked_mul(bits as usize)? / 8;
                let norms = rows.checked_mul(padded / group)?.checked_mul(2)?;
                packed.checked_add(norms)
            });
            expect("cq", needed)?;
        }
    }
    Ok(Entry { dtype, shape, offset, nbytes, group: if dtype == EntryDtype::Cq { group } else { 0 }, bits: if dtype == EntryDtype::Cq { bits } else { 0 } })
}

/// A borrowed view of one CQ matrix: logically `[rows, cols]` row-major,
/// stored as packed codebook indices plus fp16 group norms.
#[derive(Debug, Clone, Copy)]
pub struct CqMatrix<'a> {
    /// Output rows.
    pub rows: usize,
    /// Logical input columns.
    pub cols: usize,
    /// Columns after padding to whole groups.
    pub padded_cols: usize,
    /// Index width.
    pub bits: u32,
    /// Quantisation group (columns per norm).
    pub group: usize,
    /// Packed indices, `rows` rows of `padded_cols * bits / 8` bytes, LSB-first.
    pub packed: &'a [u8],
    /// fp16 norms, little-endian, `[rows, padded_cols / group]`.
    pub norms: &'a [u8],
    /// The shared codebook for `bits`.
    pub codebook: &'a [f32],
}

impl<'a> CqMatrix<'a> {
    /// Bytes of packed indices per row.
    pub fn row_bytes(&self) -> usize {
        self.padded_cols * self.bits as usize / 8
    }

    /// Groups per row.
    pub fn groups_per_row(&self) -> usize {
        self.padded_cols / self.group
    }

    /// The packed indices of `row`.
    pub fn row_packed(&self, row: usize) -> &'a [u8] {
        &self.packed[row * self.row_bytes()..(row + 1) * self.row_bytes()]
    }

    /// The norm of group `group_index` in `row`.
    pub fn norm(&self, row: usize, group_index: usize) -> f32 {
        let at = (row * self.groups_per_row() + group_index) * 2;
        half::f16::from_le_bytes([self.norms[at], self.norms[at + 1]]).to_f32()
    }

    /// Unpack `row`'s indices into `out` (`padded_cols` long).
    pub fn unpack_row(&self, row: usize, out: &mut [u8]) {
        let bits = self.bits as usize;
        let mask = (1u32 << bits) - 1;
        // Eight indices occupy exactly `bits` bytes, read as one LSB-first word.
        for (chunk, codes) in self.row_packed(row).chunks_exact(bits).zip(out.as_chunks_mut::<8>().0.iter_mut()) {
            let word = chunk.iter().rev().fold(0u32, |w, &b| (w << 8) | u32::from(b));
            for (i, code) in codes.iter_mut().enumerate() {
                *code = ((word >> (i * bits)) & mask) as u8;
            }
        }
    }

    /// Decode `row` into `out` (`cols` long).
    pub fn decode_row(&self, row: usize, out: &mut [f32]) {
        let mut indices = vec![0u8; self.padded_cols];
        self.unpack_row(row, &mut indices);
        let mut group_values = vec![0f32; self.group];
        let scale = 1.0 / (self.group as f32).sqrt();
        for (g, codes) in indices.chunks_exact(self.group).enumerate() {
            let norm = self.norm(row, g);
            for (v, &code) in group_values.iter_mut().zip(codes) {
                *v = self.codebook[usize::from(code)] * norm;
            }
            walsh_hadamard(&mut group_values);
            let start = g * self.group;
            for (o, v) in out[start.min(self.cols)..(start + self.group).min(self.cols)].iter_mut().zip(&group_values) {
                *o = v * scale;
            }
        }
    }

    /// The whole matrix as row-major `f32`.
    pub fn to_f32(&self) -> Vec<f32> {
        let mut out = vec![0f32; self.rows * self.cols];
        for (row, o) in out.chunks_exact_mut(self.cols).enumerate() {
            self.decode_row(row, o);
        }
        out
    }
}

/// In-place unnormalised natural-order Walsh-Hadamard transform (`x.len()` a power of two).
fn walsh_hadamard(x: &mut [f32]) {
    let n = x.len();
    let mut h = 1;
    while h < n {
        for block in x.chunks_exact_mut(h * 2) {
            let (a, b) = block.split_at_mut(h);
            for (l, r) in a.iter_mut().zip(b) {
                let (sum, diff) = (*l + *r, *l - *r);
                *l = sum;
                *r = diff;
            }
        }
        h *= 2;
    }
}

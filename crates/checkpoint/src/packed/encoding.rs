// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
// Swedish Embedded AB implements memory-bounded loading and training of quantized models for its clients. If your
// team needs expertise in low-precision model storage then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The vocabulary every quantized-tensor consumer shares: what a tensor's bytes
//! *mean*, independent of which file or device holds them.
//!
//! A [`TensorDescriptor`] is a logical shape plus a [`QuantEncoding`]. The
//! encoding says how the physical bytes decode to logical values:
//!
//! ```text
//! value[i] = element_value(code[i]) * scale_0[i] * scale_1[i] * ...
//! ```
//!
//! - **element**: how one code decodes ([`Element`]): an integer, a small float
//!   (E4M3, E5M2, E2M1, ...), an entry of a fixed codebook (NF4), or a
//!   self-describing GGUF block format whose own sub-block scales live inside
//!   the block ([`Element::Ggml`]).
//! - **packing**: derived from the element ([`QuantEncoding::packing`]). Dense
//!   little-endian, sub-byte codes LSB-first: element `i` of a 4-bit tensor sits
//!   in the low nibble of byte `i/2` when `i` is even, the high nibble otherwise.
//! - **scales**: a hierarchy of [`ScaleLevel`]s, each a separate sidecar tensor
//!   (named `<tensor name><suffix>`), coarse to fine. NVFP4 is
//!   E2M1 x E4M3-per-16 x f32-per-tensor ([`QuantEncoding::nvfp4`]); MXFP4 is
//!   E2M1 x E8M0-per-32 ([`QuantEncoding::mxfp4`]).
//!
//! The **logical layout** is the row-major `shape` of decoded values. The
//! **physical layout** is the payload byte stream (row-major element order,
//! packed as above) plus one sidecar scale tensor per level. A *tile* is a flat
//! element range `[start, end)` that is a multiple of
//! [`TensorDescriptor::tile_quantum`]; tiles are the unit of every bounded read,
//! decode and conversion, so no consumer ever needs the whole tensor.

use std::fmt;
use std::ops::Range;

use crate::gguf::GgmlType;

/// A small floating point format (sign/exponent/mantissa), used both for
/// elements and for scales.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FloatFormat {
    F32,
    F16,
    Bf16,
    /// OCP FP8 E4M3 ("FN": no infinities, one NaN code).
    E4M3Fn,
    /// OCP FP8 E5M2 (IEEE-like: infinities and NaNs).
    E5M2,
    /// OCP microscaling FP4: 1 sign, 2 exponent, 1 mantissa bit.
    E2M1,
    /// Unsigned power-of-two scale: `2^(code - 127)`, code 255 is NaN.
    E8M0,
}

impl FloatFormat {
    /// Bits per code.
    pub fn bits(self) -> u32 {
        match self {
            FloatFormat::F32 => 32,
            FloatFormat::F16 | FloatFormat::Bf16 => 16,
            FloatFormat::E4M3Fn | FloatFormat::E5M2 | FloatFormat::E8M0 => 8,
            FloatFormat::E2M1 => 4,
        }
    }
}

/// A fixed 16-entry codebook; the 4-bit code indexes the table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Codebook {
    /// The bitsandbytes NF4 quantile table.
    Nf4,
}

/// How one stored code decodes to a number before scaling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Element {
    /// Two's complement (`signed`) or unsigned integer of `bits` (1..=32).
    Int { bits: u8, signed: bool },
    Float(FloatFormat),
    Codebook(Codebook),
    /// A GGUF block format (Q8_0, Q4_K, MXFP4, ...). The format carries its own
    /// interleaved sub-block scales, so [`QuantEncoding::scales`] holds only
    /// *additional* outer levels (normally none).
    Ggml(GgmlType),
}

/// Which elements share one scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScaleGranularity {
    /// One scale for the whole tensor.
    Tensor,
    /// One scale per `n` consecutive elements along the innermost axis.
    Block(u32),
    /// One scale per row (all elements sharing every index but the innermost).
    Row,
    /// One scale per `rows x cols` tile of a rank-2 tensor (DeepSeek/Qwen FP8
    /// `weight_block_size`); edge tiles are partial.
    Tile2d { rows: u32, cols: u32 },
}

/// One level of the scale hierarchy: a sidecar tensor of `format` values.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ScaleLevel {
    pub granularity: ScaleGranularity,
    pub format: FloatFormat,
    /// The sidecar tensor is named `<payload name><suffix>`.
    pub suffix: String,
}

/// How payload bytes are laid out (derived from the element; see module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Packing {
    /// One element per whole number of bytes.
    ByteAligned,
    /// Sub-byte codes, little-endian, LSB-first.
    SubByteLsbFirst,
    /// Opaque fixed-size blocks defined by a GGUF type.
    GgmlBlocks,
}

/// The payload's smallest independently addressable unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockGeometry {
    pub elems: u64,
    pub bytes: u64,
}

/// The suffix of the group-scale sibling of a packed-int8 tensor in brain's int8-native checkpoints
/// (`weightio::PACKED_INT8_LAYOUT`): `<name>` holds the int8 codes as `u32` words, `<name>.scale` the f32 scales.
pub const INT8_SCALE_SUFFIX: &str = ".scale";

/// How a tensor's bytes decode to values.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct QuantEncoding {
    pub element: Element,
    /// Outer scale levels, coarse to fine. Decoded value = element value times
    /// the product of the covering scale of every level.
    pub scales: Vec<ScaleLevel>,
}

impl QuantEncoding {
    fn plain(f: FloatFormat) -> QuantEncoding {
        QuantEncoding { element: Element::Float(f), scales: Vec::new() }
    }
    pub fn f32() -> QuantEncoding {
        Self::plain(FloatFormat::F32)
    }
    pub fn f16() -> QuantEncoding {
        Self::plain(FloatFormat::F16)
    }
    pub fn bf16() -> QuantEncoding {
        Self::plain(FloatFormat::Bf16)
    }
    /// FP8 E4M3 with one bf16 multiplier per `block x block` tile, the layout of
    /// DeepSeek-V3 / Qwen3.5 `weight_scale_inv` checkpoints.
    pub fn fp8_e4m3_tiled(block: u32) -> QuantEncoding {
        QuantEncoding {
            element: Element::Float(FloatFormat::E4M3Fn),
            scales: vec![ScaleLevel { granularity: ScaleGranularity::Tile2d { rows: block, cols: block }, format: FloatFormat::Bf16, suffix: "_scale_inv".into() }],
        }
    }
    /// Unscaled FP8 (scale 1).
    pub fn fp8(format: FloatFormat) -> QuantEncoding {
        assert!(matches!(format, FloatFormat::E4M3Fn | FloatFormat::E5M2), "fp8() takes E4M3Fn or E5M2");
        Self::plain(format)
    }
    /// Signed int8 with one f32 scale per `block` elements (brain's packed int8
    /// storage uses `block = 32`; four int8 per little-endian `u32`). The scale
    /// sidecar is `<name>.scale`, the name of brain's int8-native checkpoints
    /// ([`INT8_SCALE_SUFFIX`]).
    pub fn int8_blocked(block: u32) -> QuantEncoding {
        QuantEncoding {
            element: Element::Int { bits: 8, signed: true },
            scales: vec![ScaleLevel { granularity: ScaleGranularity::Block(block), format: FloatFormat::F32, suffix: INT8_SCALE_SUFFIX.into() }],
        }
    }
    /// NF4 codes with one f32 scale per `block` elements (brain uses 32; eight
    /// codes per little-endian `u32`).
    pub fn nf4_blocked(block: u32) -> QuantEncoding {
        QuantEncoding {
            element: Element::Codebook(Codebook::Nf4),
            scales: vec![ScaleLevel { granularity: ScaleGranularity::Block(block), format: FloatFormat::F32, suffix: "_scale".into() }],
        }
    }
    /// NVFP4: E2M1 codes, an E4M3 scale per 16 elements, an f32 scale per tensor.
    pub fn nvfp4() -> QuantEncoding {
        QuantEncoding {
            element: Element::Float(FloatFormat::E2M1),
            scales: vec![
                ScaleLevel { granularity: ScaleGranularity::Tensor, format: FloatFormat::F32, suffix: "_scale_2".into() },
                ScaleLevel { granularity: ScaleGranularity::Block(16), format: FloatFormat::E4M3Fn, suffix: "_scale".into() },
            ],
        }
    }
    /// MXFP4: E2M1 codes, an E8M0 power-of-two scale per 32 elements.
    pub fn mxfp4() -> QuantEncoding {
        QuantEncoding {
            element: Element::Float(FloatFormat::E2M1),
            scales: vec![ScaleLevel { granularity: ScaleGranularity::Block(32), format: FloatFormat::E8M0, suffix: "_scale".into() }],
        }
    }
    /// A GGUF block format stored verbatim.
    pub fn ggml(ty: GgmlType) -> QuantEncoding {
        QuantEncoding { element: Element::Ggml(ty), scales: Vec::new() }
    }

    /// Payload bits per element (fractional for block formats, e.g. Q4_K = 4.5).
    pub fn payload_bits_per_elem(&self) -> f64 {
        match self.element {
            Element::Int { bits, .. } => bits as f64,
            Element::Float(f) => f.bits() as f64,
            Element::Codebook(_) => 4.0,
            Element::Ggml(t) => t.block_bytes() as f64 * 8.0 / t.block_elems() as f64,
        }
    }

    pub fn packing(&self) -> Packing {
        match self.element {
            Element::Ggml(_) => Packing::GgmlBlocks,
            _ if self.payload_bits_per_elem() < 8.0 => Packing::SubByteLsbFirst,
            _ => Packing::ByteAligned,
        }
    }

    /// The payload's smallest addressable unit: one element for byte-aligned
    /// types, `8/bits` elements per byte for sub-byte codes, one GGUF block.
    pub fn block_geometry(&self) -> BlockGeometry {
        match self.element {
            Element::Ggml(t) => BlockGeometry { elems: t.block_elems() as u64, bytes: t.block_bytes() as u64 },
            _ => {
                let bits = self.payload_bits_per_elem() as u64;
                if bits < 8 {
                    BlockGeometry { elems: 8 / bits, bytes: 1 }
                } else {
                    BlockGeometry { elems: 1, bytes: bits / 8 }
                }
            }
        }
    }
}

impl FloatFormat {
    fn tag(self) -> &'static str {
        match self {
            FloatFormat::F32 => "f32",
            FloatFormat::F16 => "f16",
            FloatFormat::Bf16 => "bf16",
            FloatFormat::E4M3Fn => "e4m3",
            FloatFormat::E5M2 => "e5m2",
            FloatFormat::E2M1 => "e2m1",
            FloatFormat::E8M0 => "e8m0",
        }
    }

    fn from_tag(t: &str) -> Option<FloatFormat> {
        [FloatFormat::F32, FloatFormat::F16, FloatFormat::Bf16, FloatFormat::E4M3Fn, FloatFormat::E5M2, FloatFormat::E2M1, FloatFormat::E8M0].into_iter().find(|f| f.tag() == t)
    }
}

impl QuantEncoding {
    /// The encoding as a self-describing string a checkpoint file can carry in its metadata
    /// (`element;level;level...`, e.g. `codebook:nf4;block:32:f32:_scale`), so a reader needs no per-model
    /// knowledge to interpret it. `None` for a GGUF block format, which a GGUF file already describes.
    pub fn spec(&self) -> Option<String> {
        let element = match self.element {
            Element::Int { bits, signed } => format!("int:{bits}:{}", if signed { "s" } else { "u" }),
            Element::Float(f) => format!("float:{}", f.tag()),
            Element::Codebook(Codebook::Nf4) => "codebook:nf4".to_string(),
            Element::Ggml(_) => return None,
        };
        let mut out = element;
        for s in &self.scales {
            let g = match s.granularity {
                ScaleGranularity::Tensor => "tensor".to_string(),
                ScaleGranularity::Block(n) => format!("block:{n}"),
                ScaleGranularity::Row => "row".to_string(),
                ScaleGranularity::Tile2d { rows, cols } => format!("tile:{rows}x{cols}"),
            };
            out.push_str(&format!(";{g}:{}:{}", s.format.tag(), s.suffix));
        }
        Some(out)
    }

    /// Inverse of [`Self::spec`].
    pub fn from_spec(spec: &str) -> Result<QuantEncoding, String> {
        let bad = || format!("unreadable quantization spec {spec:?}");
        let mut parts = spec.split(';');
        let element = parts.next().ok_or_else(bad)?;
        let e: Vec<&str> = element.split(':').collect();
        let element = match e.as_slice() {
            ["int", bits, sign] => Element::Int { bits: bits.parse().map_err(|_| bad())?, signed: *sign == "s" },
            ["float", f] => Element::Float(FloatFormat::from_tag(f).ok_or_else(bad)?),
            ["codebook", "nf4"] => Element::Codebook(Codebook::Nf4),
            _ => return Err(bad()),
        };
        let mut scales = Vec::new();
        for level in parts {
            let l: Vec<&str> = level.splitn(4, ':').collect();
            let (granularity, fmt, suffix) = match l.as_slice() {
                ["tensor", fmt, suffix] => (ScaleGranularity::Tensor, fmt, suffix),
                ["row", fmt, suffix] => (ScaleGranularity::Row, fmt, suffix),
                ["block", n, fmt, suffix] => (ScaleGranularity::Block(n.parse().map_err(|_| bad())?), fmt, suffix),
                ["tile", rc, fmt, suffix] => {
                    let (r, c) = rc.split_once('x').ok_or_else(bad)?;
                    (ScaleGranularity::Tile2d { rows: r.parse().map_err(|_| bad())?, cols: c.parse().map_err(|_| bad())? }, fmt, suffix)
                }
                _ => return Err(bad()),
            };
            scales.push(ScaleLevel { granularity, format: FloatFormat::from_tag(fmt).ok_or_else(bad)?, suffix: suffix.to_string() });
        }
        Ok(QuantEncoding { element, scales })
    }
}

impl fmt::Display for QuantEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.element {
            Element::Int { bits, signed } => write!(f, "{}{bits}", if signed { "i" } else { "u" })?,
            Element::Float(x) => write!(f, "{x:?}")?,
            Element::Codebook(c) => write!(f, "{c:?}")?,
            Element::Ggml(t) => write!(f, "{}", t.name())?,
        }
        for s in &self.scales {
            match s.granularity {
                ScaleGranularity::Tensor => write!(f, " x {:?}/tensor", s.format)?,
                ScaleGranularity::Block(n) => write!(f, " x {:?}/{n}", s.format)?,
                ScaleGranularity::Row => write!(f, " x {:?}/row", s.format)?,
                ScaleGranularity::Tile2d { rows, cols } => write!(f, " x {:?}/{rows}x{cols}", s.format)?,
            }
        }
        Ok(())
    }
}

/// A named tensor: logical shape plus encoding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorDescriptor {
    pub name: String,
    /// Logical (decoded) row-major shape.
    pub shape: Vec<u64>,
    pub encoding: QuantEncoding,
}

fn lcm(a: u64, b: u64) -> u64 {
    fn gcd(a: u64, b: u64) -> u64 {
        if b == 0 { a } else { gcd(b, a % b) }
    }
    a / gcd(a, b) * b
}

impl TensorDescriptor {
    pub fn new(name: impl Into<String>, shape: Vec<u64>, encoding: QuantEncoding) -> TensorDescriptor {
        TensorDescriptor { name: name.into(), shape, encoding }
    }

    pub fn numel(&self) -> u64 {
        self.shape.iter().product()
    }

    /// Size of the innermost axis (1 for a scalar).
    fn inner(&self) -> u64 {
        self.shape.last().copied().unwrap_or(1)
    }

    /// Product of every axis but the innermost.
    fn rows(&self) -> u64 {
        self.numel() / self.inner().max(1)
    }

    /// Check that the shape and encoding are mutually consistent: the payload
    /// must be a whole number of blocks, block scales must not straddle rows,
    /// and tile scales need a rank-2 tensor.
    pub fn validate(&self) -> Result<(), String> {
        let name = &self.name;
        let g = self.encoding.block_geometry();
        if self.numel() == 0 {
            return Err(format!("{name}: empty tensor {:?}", self.shape));
        }
        if !self.numel().is_multiple_of(g.elems) {
            return Err(format!("{name}: {} elements is not a multiple of the {}-element payload block of {}", self.numel(), g.elems, self.encoding));
        }
        if matches!(self.encoding.element, Element::Ggml(_)) && !self.inner().is_multiple_of(g.elems) {
            return Err(format!("{name}: innermost axis {} is not a multiple of the {}-element block of {}", self.inner(), g.elems, self.encoding));
        }
        for s in &self.encoding.scales {
            match s.granularity {
                ScaleGranularity::Block(n) if n == 0 || !self.inner().is_multiple_of(n as u64) => {
                    return Err(format!("{name}: innermost axis {} is not a multiple of scale block {n}", self.inner()));
                }
                ScaleGranularity::Tile2d { rows, cols } if rows == 0 || cols == 0 || self.shape.len() != 2 => {
                    return Err(format!("{name}: {rows}x{cols} tile scales need a rank-2 tensor, got shape {:?}", self.shape));
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Payload size in bytes.
    pub fn payload_bytes(&self) -> u64 {
        let g = self.encoding.block_geometry();
        self.numel().div_ceil(g.elems) * g.bytes
    }

    /// Number of scale values at `level`.
    pub fn scale_numel(&self, level: usize) -> u64 {
        match self.encoding.scales[level].granularity {
            ScaleGranularity::Tensor => 1,
            ScaleGranularity::Block(n) => self.numel() / n as u64,
            ScaleGranularity::Row => self.rows(),
            ScaleGranularity::Tile2d { rows, cols } => self.shape[0].div_ceil(rows as u64) * self.shape[1].div_ceil(cols as u64),
        }
    }

    /// Sidecar size in bytes at `level`.
    pub fn scale_bytes(&self, level: usize) -> u64 {
        self.scale_numel(level) * self.encoding.scales[level].format.bits() as u64 / 8
    }

    /// Smallest legal tile length: tile starts and lengths (except a final
    /// partial tile) must be multiples of it. Whole rows for row and 2-D tile
    /// scales, so a tile's scales form one contiguous sidecar range.
    pub fn tile_quantum(&self) -> u64 {
        let mut q = self.encoding.block_geometry().elems;
        for s in &self.encoding.scales {
            q = match s.granularity {
                ScaleGranularity::Tensor => q,
                ScaleGranularity::Block(n) => lcm(q, n as u64),
                ScaleGranularity::Row | ScaleGranularity::Tile2d { .. } => lcm(q, self.inner()),
            };
        }
        q
    }

    /// Smallest tile an encoder can produce exactly: [`Self::tile_quantum`], and whole scale bands
    /// for 2-D tile scales (a scale is computed from all of its tile's values at once).
    pub fn encode_quantum(&self) -> u64 {
        let mut q = self.tile_quantum();
        for s in &self.encoding.scales {
            if let ScaleGranularity::Tile2d { rows, .. } = s.granularity {
                q = lcm(q, rows as u64 * self.inner());
            }
        }
        q
    }

    /// Payload byte range covering tile `elems` (`start` must be block aligned).
    pub fn payload_range(&self, elems: &Range<u64>) -> Range<u64> {
        let g = self.encoding.block_geometry();
        debug_assert!(elems.start.is_multiple_of(g.elems), "tile start not block aligned");
        elems.start / g.elems * g.bytes..elems.end.div_ceil(g.elems) * g.bytes
    }

    /// Range of `level`'s scale values covering tile `elems` (tile aligned to
    /// [`Self::tile_quantum`]).
    pub fn scale_range(&self, level: usize, elems: &Range<u64>) -> Range<u64> {
        match self.encoding.scales[level].granularity {
            ScaleGranularity::Tensor => 0..1,
            ScaleGranularity::Block(n) => elems.start / n as u64..elems.end.div_ceil(n as u64),
            ScaleGranularity::Row => elems.start / self.inner()..elems.end.div_ceil(self.inner()),
            ScaleGranularity::Tile2d { rows, cols } => {
                let band = |row: u64| row / rows as u64;
                let r0 = elems.start / self.inner();
                let r1 = elems.end.div_ceil(self.inner());
                let per_band = self.shape[1].div_ceil(cols as u64);
                band(r0) * per_band..(band(r1 - 1) + 1) * per_band
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(shape: &[u64], e: QuantEncoding) -> TensorDescriptor {
        TensorDescriptor::new("w", shape.to_vec(), e)
    }

    #[test]
    fn every_non_ggml_encoding_round_trips_through_its_spec() {
        for e in [QuantEncoding::f32(), QuantEncoding::bf16(), QuantEncoding::fp8_e4m3_tiled(128), QuantEncoding::int8_blocked(32), QuantEncoding::nf4_blocked(32), QuantEncoding::nvfp4(), QuantEncoding::mxfp4(), QuantEncoding::fp8(FloatFormat::E5M2)] {
            let spec = e.spec().expect("a non-ggml encoding has a spec");
            assert_eq!(QuantEncoding::from_spec(&spec).unwrap(), e, "{spec}");
        }
        assert!(QuantEncoding::ggml(GgmlType::Q8_0).spec().is_none());
        assert!(QuantEncoding::from_spec("float:nonsense").is_err() && QuantEncoding::from_spec("codebook:nf4;block:x:f32:_s").is_err());
    }

    #[test]
    fn nvfp4_is_e2m1_with_block16_e4m3_and_global_f32_scales() {
        let d = desc(&[4096, 4096], QuantEncoding::nvfp4());
        d.validate().unwrap();
        let n = 4096u64 * 4096;
        assert_eq!(d.payload_bytes(), n / 2);
        assert_eq!((d.scale_numel(0), d.scale_bytes(0)), (1, 4));
        assert_eq!((d.scale_numel(1), d.scale_bytes(1)), (n / 16, n / 16));
        assert_eq!(d.encoding.payload_bits_per_elem(), 4.0);
        assert_eq!(d.encoding.packing(), Packing::SubByteLsbFirst);
    }

    #[test]
    fn mxfp4_is_e2m1_with_e8m0_block32_scales() {
        let d = desc(&[64, 128], QuantEncoding::mxfp4());
        d.validate().unwrap();
        assert_eq!(d.payload_bytes(), 64 * 128 / 2);
        assert_eq!(d.scale_bytes(0), 64 * 128 / 32);
    }

    #[test]
    fn fp8_tiles_round_up_at_the_edges_and_a_tile_of_rows_maps_to_whole_scale_bands() {
        let d = desc(&[300, 130], QuantEncoding::fp8_e4m3_tiled(128));
        d.validate().unwrap();
        assert_eq!(d.scale_numel(0), 3 * 2);
        assert_eq!(d.scale_bytes(0), 12);
        assert_eq!(d.tile_quantum(), 130);
        // rows 100..140 straddle bands 0 and 1.
        assert_eq!(d.scale_range(0, &(100 * 130..140 * 130)), 0..4);
        assert_eq!(d.scale_range(0, &(256 * 130..300 * 130)), 4..6);
    }

    #[test]
    fn ggml_geometry_is_the_gguf_block_table() {
        for t in [GgmlType::Q8_0, GgmlType::Q4K, GgmlType::Q6K, GgmlType::MXFP4] {
            let d = desc(&[2, 2 * QK], QuantEncoding::ggml(t));
            let g = d.encoding.block_geometry();
            assert_eq!((g.elems as usize, g.bytes as usize), (t.block_elems(), t.block_bytes()));
            assert_eq!(d.payload_bytes() as usize, crate::gguf::tensor_nbytes(t.id(), d.numel() as usize).unwrap());
        }
    }
    const QK: u64 = 256;

    #[test]
    fn tile_quantum_and_ranges_follow_block_boundaries() {
        let d = desc(&[8, 64], QuantEncoding::nvfp4());
        assert_eq!(d.tile_quantum(), 16);
        let t = 32..96;
        assert_eq!(d.payload_range(&t), 16..48);
        assert_eq!(d.scale_range(0, &t), 0..1);
        assert_eq!(d.scale_range(1, &t), 2..6);
    }

    #[test]
    fn validate_rejects_geometry_that_cannot_be_decoded() {
        assert!(desc(&[4, 30], QuantEncoding::nvfp4()).validate().is_err(), "scale block straddles the row");
        assert!(desc(&[2, 3, 128], QuantEncoding::fp8_e4m3_tiled(128)).validate().is_err(), "tile scales need rank 2");
        assert!(desc(&[3, 100], QuantEncoding::ggml(GgmlType::Q8_0)).validate().is_err(), "partial ggml block");
    }
}

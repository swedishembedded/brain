// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Compact KV-cache tiers for batched paged decode: how a GQA layer's K and V
//! planes are stored (`f32`, `bf16`, or per-row scaled `int8` / `fp8` E4M3), and
//! the kernels that append to, and attend over, a plane of each tier.
//!
//! Swedish Embedded AB implements long-context inference serving for clients
//! whose GPU memory, not their compute, decides how many concurrent users one
//! card carries. If your team needs expertise in fitting more sequences into
//! the same device memory without giving up accuracy, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! # Why a tier is a storage choice, not a model choice
//!
//! A hybrid decoder's full-attention layers keep `2 * cap * n_kv_heads *
//! head_dim` elements per sequence, which is what bounds the batch at a long
//! context (16 GiB per sequence at 128k tokens in `f32` for Qwen3.8-27B). The
//! attention arithmetic stays `f32` in every tier - only the resident bytes
//! narrow, and each element is widened as the kernel loads it:
//!
//! | tier   | bytes per element | layout                                              |
//! |--------|-------------------|-----------------------------------------------------|
//! | `f32`  | 4                 | one word per element                                |
//! | `bf16` | 2                 | two per `u32` word, the top half of an `f32`        |
//! | `int8` | 1 + 4 / head_dim  | four per `u32` word, one `f32` scale per (token, kv-head) row |
//! | `fp8`  | 1 + 4 / head_dim  | E4M3 bytes, packed and scaled exactly like `int8` (scale = absmax / 448) |
//!
//! A tier is a descriptor, [`KvTier`] = ([`KvElement`], [`KvScale`]): the element
//! codec and whether each (token, kv-head) row carries an `f32` scale. It is the
//! `Copy` handle for the subset of the checkpoint crate's [`QuantEncoding`] the
//! attention kernels can decode ([`KvTier::encoding`] / [`KvTier::from_encoding`]),
//! and every size derives from that encoding over the plane viewed as a
//! `[rows * n_kv_heads, head_dim]` tensor ([`KvTier::plane_descriptor`]), so a
//! planner that reasons about encodings and one that reasons about KV tiers
//! cannot disagree on bytes. Only the kernels differ per element codec.
//!
//! The kernels are the repo's existing paged-KV family - nothing here is a
//! second implementation of attention:
//!
//! * `bf16`: `paged_kv_append_batched_word` (packing store) and the
//!   `paged_decode_{scores,apply}_batched` loads, both rewritten by
//!   [`kernels::template`]'s bf16 variants;
//! * `int8`: `paged_kv_append_i8_clipped_batched`, `paged_decode_scores_i8_batched`
//!   and `paged_decode_apply_i8_batched`;
//! * `fp8`: `paged_kv_append_fp8_clipped_batched` and the f32 scores/apply/prefill
//!   kernels rewritten by [`kernels::template::fp8_kv_variant`];
//! * the fused `head_dim = 256` prefill kernel, whose bf16, int8 and fp8 loads are
//!   template variants of the one `f32` source.
//!
//! [`kernel_list`] is what a model registers on its [`Gpu`] to run any tier;
//! [`KvKernels::resolve`] looks the chosen tier's kernels up BY NAME.
//!
//! # Addressing limit
//!
//! The kernels index a pool with `u32` element offsets, so one plane may hold
//! at most [`MAX_PLANE_ELEMS`] (2^32) elements ([`KvTier::fits_addressing`]). At
//! `kv_stride = 1024` that is 4 194 304 token rows - 32 sequences of 128k.

use checkpoint::packed::{Element, FloatFormat, QuantEncoding, ScaleGranularity, ScaleLevel, TensorDescriptor};
use gpu_core::{DeviceBuffer, Dispatch, Gpu, Step};

use crate::ops::PagedDecodeShape;

/// Largest number of elements one plane may hold: the kernels' `u32` element
/// index reaches `0..2^32`. Past it a pool would silently alias its own start.
pub const MAX_PLANE_ELEMS: u64 = 1 << 32;

/// How one KV element is encoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KvElement {
    F32,
    Bf16,
    /// Signed 8-bit integer, symmetric around zero (`-127..=127`).
    Int8,
    /// 8-bit float, 4 exponent and 3 mantissa bits (OCP FP8 E4M3, max 448).
    Fp8E4m3,
}

/// The scale hierarchy of a plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KvScale {
    /// Elements are stored at their own magnitude.
    None,
    /// One `f32` scale per (token, kv-head) row of `head_dim` elements.
    PerRow,
}

/// How a KV plane is stored: an element codec and a scale hierarchy. The
/// associated constants name the supported tiers; see the module doc for layouts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KvTier {
    element: KvElement,
    scale: KvScale,
}

/// Device words one plane of a tier allocates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneWords {
    /// The element storage.
    pub data: u64,
    /// The per-(token, kv-head) scales; `0` for tiers without any.
    pub scales: u64,
}

impl PlaneWords {
    /// Device bytes: four per word.
    pub fn bytes(self) -> u64 {
        (self.data + self.scales) * 4
    }
}

impl KvTier {
    pub const F32: KvTier = KvTier { element: KvElement::F32, scale: KvScale::None };
    pub const BF16: KvTier = KvTier { element: KvElement::Bf16, scale: KvScale::None };
    pub const INT8: KvTier = KvTier { element: KvElement::Int8, scale: KvScale::PerRow };
    pub const FP8: KvTier = KvTier { element: KvElement::Fp8E4m3, scale: KvScale::PerRow };

    /// Every tier, in order of decreasing size.
    pub const ALL: [KvTier; 4] = [KvTier::F32, KvTier::BF16, KvTier::INT8, KvTier::FP8];

    pub fn element(self) -> KvElement {
        self.element
    }

    pub fn scale(self) -> KvScale {
        self.scale
    }

    /// The checkpoint-crate encoding of this tier's bytes: the element codec plus,
    /// for a scaled tier, one `f32` scale per row of the `[rows * n_kv, head_dim]`
    /// view of a plane.
    pub fn encoding(self) -> QuantEncoding {
        let element = match self.element {
            KvElement::F32 => Element::Float(FloatFormat::F32),
            KvElement::Bf16 => Element::Float(FloatFormat::Bf16),
            KvElement::Int8 => Element::Int { bits: 8, signed: true },
            KvElement::Fp8E4m3 => Element::Float(FloatFormat::E4M3Fn),
        };
        let scales = match self.scale {
            KvScale::None => Vec::new(),
            KvScale::PerRow => vec![ScaleLevel { granularity: ScaleGranularity::Row, format: FloatFormat::F32, suffix: "_scale".into() }],
        };
        QuantEncoding { element, scales }
    }

    /// The tier that stores exactly `enc`, or why none does (the kernels decode
    /// f32, bf16, symmetric int8 and E4M3 elements, scaled per row or not at all).
    pub fn from_encoding(enc: &QuantEncoding) -> Result<KvTier, String> {
        KvTier::ALL.into_iter().find(|t| t.encoding() == *enc).ok_or_else(|| format!("no KV tier stores {enc}: supported are {}", KvTier::ALL.map(|t| t.encoding().to_string()).join(", ")))
    }

    /// A plane of `rows` token rows as the tensor `[rows * n_kv_heads, head_dim]` this
    /// tier's encoding describes.
    pub fn plane_descriptor(self, rows: u64, kv_stride: u64, head_dim: u64) -> TensorDescriptor {
        TensorDescriptor::new("kv_plane", vec![rows * (kv_stride / head_dim), head_dim], self.encoding())
    }

    /// Elements packed into one `u32` word.
    fn per_word(self) -> u64 {
        32 / self.encoding().payload_bits_per_elem() as u64
    }

    /// Parse a user-facing name: `f32`/`fp32`, `bf16`, `int8`/`i8`, `fp8`/`e4m3`/`fp8-e4m3`.
    pub fn parse(s: &str) -> Result<KvTier, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f32" | "fp32" => Ok(KvTier::F32),
            "bf16" => Ok(KvTier::BF16),
            "int8" | "i8" => Ok(KvTier::INT8),
            "fp8" | "e4m3" | "fp8-e4m3" | "fp8e4m3" => Ok(KvTier::FP8),
            other => Err(format!("unknown KV tier {other:?}: expected f32, bf16, int8 or fp8")),
        }
    }

    /// The name [`Self::parse`] accepts, and the one reports print.
    pub fn as_str(self) -> &'static str {
        match self.element {
            KvElement::F32 => "f32",
            KvElement::Bf16 => "bf16",
            KvElement::Int8 => "int8",
            KvElement::Fp8E4m3 => "fp8",
        }
    }

    /// The tier named by environment variable `var`; unset or empty is
    /// [`KvTier::F32`]. An unparseable value is an error, never a silent
    /// fallback: serving a different precision than was asked for is worse
    /// than refusing to start.
    pub fn from_env(var: &str) -> Result<KvTier, String> {
        match std::env::var(var) {
            Ok(s) if !s.trim().is_empty() => Self::parse(&s).map_err(|e| format!("{var}: {e}")),
            _ => Ok(KvTier::F32),
        }
    }

    /// The device words one K (or V) plane of `rows` token rows allocates.
    /// `kv_stride = n_kv_heads * head_dim`.
    pub fn plane_words(self, rows: u64, kv_stride: u64, head_dim: u64) -> PlaneWords {
        if rows == 0 {
            return PlaneWords { data: 0, scales: 0 };
        }
        let d = self.plane_descriptor(rows, kv_stride, head_dim);
        let scale_bytes = if d.encoding.scales.is_empty() { 0 } else { d.scale_bytes(0) };
        PlaneWords { data: d.payload_bytes().div_ceil(4), scales: scale_bytes / 4 }
    }

    /// Device bytes of one K (or V) plane of `rows` token rows.
    pub fn plane_bytes(self, rows: u64, kv_stride: u64, head_dim: u64) -> u64 {
        self.plane_words(rows, kv_stride, head_dim).bytes()
    }

    /// Whether a plane of `rows` token rows can be addressed by the kernels'
    /// `u32` element offsets.
    pub fn fits_addressing(rows: u64, kv_stride: u64) -> bool {
        rows * kv_stride <= MAX_PLANE_ELEMS
    }
}

impl std::fmt::Display for KvTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Host reference decode of one FP8 E4M3 byte (sign, 4 exponent bits with bias 7,
/// 3 mantissa bits; exponent 0 is subnormal). `0x7F` and `0xFF` are NaN in the
/// format; the append kernel never writes them.
pub fn decode_e4m3(byte: u8) -> f32 {
    let (e, m) = (i32::from(byte >> 3 & 15), f32::from(byte & 7));
    let mag = if e == 0 { m * 2f32.powi(-9) } else { (1.0 + m / 8.0) * 2f32.powi(e - 7) };
    if byte & 0x80 != 0 { -mag } else { mag }
}

/// One K or V plane of one layer: `rows` token rows in a [`KvTier`].
pub struct KvPlane {
    tier: KvTier,
    /// The element storage, laid out per tier (module doc).
    data: DeviceBuffer,
    /// `[rows, n_kv_heads]` row scales of the int8 tier.
    scales: Option<DeviceBuffer>,
    /// `[n_kv_heads]` calibrated append ceiling of the int8 tier, `f32::MAX`:
    /// no calibration, so the append kernel's scale is the row's own absmax.
    clip: Option<DeviceBuffer>,
    /// Entries of `clip` (kv heads); `0` for tiers without one.
    n_kv: usize,
}

impl KvPlane {
    /// A plane of `rows` token rows of `kv_stride = n_kv_heads * head_dim`
    /// elements. Panics if the plane cannot be addressed ([`MAX_PLANE_ELEMS`])
    /// or `kv_stride` is not a whole number of heads, or - for a
    /// per-row scaled tier - the head is not a whole number of packed words.
    pub fn new(gpu: &Gpu, tier: KvTier, rows: u64, kv_stride: u64, head_dim: u64) -> KvPlane {
        assert!(head_dim > 0 && kv_stride.is_multiple_of(head_dim), "KvPlane: kv_stride {kv_stride} is not a whole number of {head_dim}-wide heads");
        assert!(KvTier::fits_addressing(rows, kv_stride), "KvPlane: {rows} rows x {kv_stride} elements exceed the kernels' u32 element index ({MAX_PLANE_ELEMS})");
        let words = tier.plane_words(rows, kv_stride, head_dim);
        match tier.scale {
            KvScale::None => KvPlane { tier, data: gpu.storage(words.data), scales: None, clip: None, n_kv: 0 },
            KvScale::PerRow => {
                assert!(head_dim.is_multiple_of(tier.per_word()), "KvPlane: a {tier} head of {head_dim} elements is not a whole number of packed words");
                let n_kv = (kv_stride / head_dim) as usize;
                KvPlane {
                    tier,
                    data: gpu.storage(words.data),
                    scales: Some(gpu.storage(words.scales)),
                    clip: Some(gpu.storage_init("kv.clip", &vec![f32::MAX; n_kv])),
                    n_kv,
                }
            }
        }
    }

    /// A one-word `f32` stand-in for a layer that keeps no KV (a GDN layer, or
    /// one another shard owns): every layer index carries a plane, so callers
    /// never branch on layer type to index.
    pub fn placeholder(gpu: &Gpu) -> KvPlane {
        KvPlane { tier: KvTier::F32, data: gpu.storage(1), scales: None, clip: None, n_kv: 0 }
    }

    pub fn tier(&self) -> KvTier {
        self.tier
    }

    /// Install per-kv-head calibrated append ceilings (`[n_kv]`, see
    /// [`crate::kvcalib`]) on a scaled plane; `f32::MAX` leaves a head uncalibrated.
    /// Returns `false`, changing nothing, for a tier without row scales.
    pub fn set_clip(&mut self, gpu: &Gpu, ceilings: &[f32]) -> bool {
        match &self.clip {
            Some(buf) => {
                assert_eq!(ceilings.len(), self.n_kv, "KvPlane::set_clip: {} ceilings for a plane with a different head count", ceilings.len());
                gpu.write_f32(buf, ceilings);
                true
            }
            None => false,
        }
    }

    /// The plane's first `rows` rows decoded to `f32` on the host, `[rows,
    /// kv_stride]`: what an attention kernel sees, for diagnostics and for
    /// tests that compare a tier's loads against `f32` ones over the same
    /// values. Reads the whole plane back - never on a serving path.
    pub fn read_dequantized(&self, gpu: &Gpu, rows: usize, kv_stride: usize, head_dim: usize) -> Vec<f32> {
        let elems = rows * kv_stride;
        let word_elems = self.tier.per_word() as usize;
        let words = gpu.read(&self.data, elems.div_ceil(word_elems));
        let elem_bits = 32 / word_elems;
        let bits_of = |i: usize| (words[i / word_elems].to_bits() >> (elem_bits * (i % word_elems))) & (u32::MAX >> (32 - elem_bits));
        let scale_of = |i: usize, scales: &[f32]| scales[i / head_dim];
        let scales = self.scales.as_ref().map(|sc| gpu.read(sc, rows * (kv_stride / head_dim)));
        (0..elems)
            .map(|i| {
                let raw = bits_of(i);
                match self.tier.element {
                    KvElement::F32 => f32::from_bits(raw),
                    KvElement::Bf16 => f32::from_bits(raw << 16),
                    KvElement::Int8 => f32::from(raw as u8 as i8) * scale_of(i, scales.as_ref().expect("int8 plane has scales")),
                    KvElement::Fp8E4m3 => decode_e4m3(raw as u8) * scale_of(i, scales.as_ref().expect("fp8 plane has scales")),
                }
            })
            .collect()
    }

    /// The element storage, for kernels outside this module that read an `f32`
    /// plane directly.
    pub fn data(&self) -> &DeviceBuffer {
        &self.data
    }

    /// `[rows, n_kv_heads]` row scales of a scaled (int8, fp8) plane, for the fused decode prep that appends in place.
    pub fn scales(&self) -> Option<&DeviceBuffer> {
        self.scales.as_ref()
    }

    /// `[n_kv_heads]` calibrated append ceilings of a scaled plane.
    pub fn clip(&self) -> Option<&DeviceBuffer> {
        self.clip.as_ref()
    }

    /// Steps that copy token rows `from_row .. from_row + rows` of this plane
    /// over rows `to_row ..` of the SAME plane - element storage and, in the
    /// int8 tier, the per-row scales - so a sequence's cached prefix can seed
    /// another sequence's window. `copy_within` is the pipeline index of the
    /// `copy_within` kernel on `gpu`: the plane is bound once, because a buffer
    /// bound as both source and destination of one dispatch is a wgpu usage
    /// conflict. The ranges must not overlap. Copies words, never values, so
    /// every tier is bit-exact.
    pub fn copy_rows(&self, gpu: &Gpu, copy_within: usize, from_row: u64, to_row: u64, rows: u64, kv_stride: u64, head_dim: u64) -> Vec<Step> {
        let mut steps = Vec::new();
        let mut copy = |buf: &DeviceBuffer, per_row: u64| {
            let n = rows * per_row;
            if n > 0 {
                steps.push(copy_words_within(gpu, copy_within, buf, from_row * per_row, to_row * per_row, n));
            }
        };
        let word_elems = self.tier.per_word();
        // Rows are word-aligned only when a row is a whole number of words.
        assert!(kv_stride.is_multiple_of(word_elems), "KvPlane::copy_rows: a {}-element row is not a whole number of {} words", kv_stride, self.tier);
        let per_row = kv_stride / word_elems;
        copy(&self.data, per_row);
        if let Some(scales) = &self.scales {
            copy(scales, kv_stride / head_dim);
        }
        steps
    }
}

/// One `copy_within` dispatch: `n` words of `buf` from word `from` to word `to`, which
/// must not overlap (an overlapping copy would race on a GPU).
pub fn copy_words_within(gpu: &Gpu, copy_within: usize, buf: &DeviceBuffer, from: u64, to: u64, n: u64) -> Step {
    assert!(from + n <= to || to + n <= from, "copy_words_within: words {from}..{} and {to}..{} overlap", from + n, to + n);
    let word = |v: u64| u32::try_from(v).unwrap_or_else(|_| panic!("copy_words_within: word offset {v} does not fit the kernel's 32-bit index"));
    gpu.dispatch(copy_within, &[buf], &[word(n), word(from), word(to)], Dispatch::Threads(word(n)))
}

/// One layer's K and V planes.
pub struct KvLayer {
    pub k: KvPlane,
    pub v: KvPlane,
}

impl KvLayer {
    /// Let a decode stream and a prefill stream write this layer's pool at once,
    /// in the disjoint rows of their own sequences (see `Gpu::share_rows`).
    pub fn share_rows(&self, gpu: &Gpu) {
        for plane in [&self.k, &self.v] {
            gpu.share_rows(&plane.data);
            if let Some(scales) = &plane.scales {
                gpu.share_rows(scales);
            }
        }
    }

    /// Both planes of a layer that keeps `rows` token rows.
    pub fn new(gpu: &Gpu, tier: KvTier, rows: u64, kv_stride: u64, head_dim: u64) -> KvLayer {
        KvLayer { k: KvPlane::new(gpu, tier, rows, kv_stride, head_dim), v: KvPlane::new(gpu, tier, rows, kv_stride, head_dim) }
    }

    /// A layer with no KV - see [`KvPlane::placeholder`].
    pub fn placeholder(gpu: &Gpu) -> KvLayer {
        KvLayer { k: KvPlane::placeholder(gpu), v: KvPlane::placeholder(gpu) }
    }
}

// ------------------------------------------------------------------ kernels

/// Kernel names per tier, in [`Names`] field order.
struct Names {
    append: &'static str,
    scores: &'static str,
    apply: &'static str,
    prefill_hd256: &'static str,
}

/// The tier-independent merge pass of the fused decode.
const FLASH_COMBINE: &str = "paged_flash_decode_gqa_combine";

/// The `(head_dim, GQA group)` pairs (group = query heads per kv head) the fused decode is
/// specialised for - its loops over a group's heads are unrolled into
/// registers, so each pair is its own pipeline. Group 6 at head_dim 256 is Qwen3.8-27B
/// (24 heads / 4 kv), group 8 the 35B-A3B sibling (16 / 2); head_dim 128 with group 1 is plain
/// multi-head attention (DeepSeek-OCR's decoder: 10 heads of 128). Any other shape keeps the triad.
const FLASH_SHAPES: [(u32, u32); 4] = [(256, 6), (256, 8), (256, 4), (128, 1)];

/// The compile-time constants of the fused decode kernel for one shape: `GRP`, and for the
/// non-native head width `HD` and its 4-channel piece count `HD4`.
fn decode_params(head_dim: u32, group: u32) -> Vec<(&'static str, u32)> {
    if head_dim == 256 {
        vec![("GRP", group)]
    } else {
        vec![("GRP", group), ("HD", head_dim), ("HD4", head_dim / 4)]
    }
}

/// The pipeline name of the fused decode for `tier` at `head_dim` and GQA group `group`:
/// the constants' specialisation first, then the tier's storage rewrite - the names
/// `kernels::template::interned` and `kv_tier_variant` give them.
fn decode_name(tier: KvTier, head_dim: u32, group: u32) -> String {
    let base = kernels::template::variant_name("paged_flash_decode_gqa_hd256", &decode_params(head_dim, group));
    match tier.element {
        KvElement::F32 => base,
        KvElement::Bf16 => format!("{base}#kv=bf16"),
        KvElement::Int8 => format!("{base}#kv=int8"),
        KvElement::Fp8E4m3 => format!("{base}#kv=fp8"),
    }
}

const F32_NAMES: Names = Names {
    append: "paged_kv_append_batched",
    scores: "paged_decode_scores_batched",
    apply: "paged_decode_apply_batched",
    prefill_hd256: "paged_flash_prefill_hd256",
};

const BF16_NAMES: Names = Names {
    append: "paged_kv_append_batched_word#pool=bf16",
    scores: "paged_decode_scores_batched#pool_k=bf16",
    apply: "paged_decode_apply_batched#pool_v=bf16",
    prefill_hd256: "paged_flash_prefill_hd256#pool_k=bf16#pool_v=bf16",
};

const INT8_NAMES: Names = Names {
    append: "paged_kv_append_i8_clipped_batched",
    scores: "paged_decode_scores_i8_batched",
    apply: "paged_decode_apply_i8_batched",
    prefill_hd256: "paged_flash_prefill_hd256#pool_k=kv8#pool_v=kv8",
};

const FP8_NAMES: Names = Names {
    append: "paged_kv_append_fp8_clipped_batched",
    scores: "paged_decode_scores_batched#pool_k=kvfp8",
    apply: "paged_decode_apply_batched#pool_v=kvfp8",
    prefill_hd256: "paged_flash_prefill_hd256#pool_k=kvfp8#pool_v=kvfp8",
};

impl KvTier {
    fn names(self) -> &'static Names {
        match self.element {
            KvElement::F32 => &F32_NAMES,
            KvElement::Bf16 => &BF16_NAMES,
            KvElement::Int8 => &INT8_NAMES,
            KvElement::Fp8E4m3 => &FP8_NAMES,
        }
    }
}

/// Every `(name, wgsl source)` pair [`KvKernels::resolve`] can ask a [`Gpu`]
/// for, across all tiers - what a model that offers the compact tiers adds to
/// its own pipeline list. Names already in `existing` are not repeated.
pub fn kernel_list(existing: &[(&str, &str)]) -> Vec<(&'static str, &'static str)> {
    use kernels::template::{dtype_variant, dtype_variant_store, fp8_kv_variant, int8_kv_variant, interned, kv_tier_variant, CompactKv};
    use gpu_core::select::Dtype;

    let have: std::collections::HashSet<&str> = existing.iter().map(|(n, _)| *n).collect();
    let bf16_prefill_k = dtype_variant("paged_flash_prefill_hd256", kernels::PAGED_FLASH_PREFILL_HD256, "pool_k", Dtype::BF16).expect("pool_k is templatable");
    let bf16_prefill = dtype_variant(bf16_prefill_k.0, bf16_prefill_k.1, "pool_v", Dtype::BF16).expect("pool_v is templatable");
    let i8_prefill_k = int8_kv_variant("paged_flash_prefill_hd256", kernels::PAGED_FLASH_PREFILL_HD256, "pool_k", "k_scales", "p.head_dim").expect("pool_k is templatable");
    let i8_prefill = int8_kv_variant(i8_prefill_k.0, i8_prefill_k.1, "pool_v", "v_scales", "p.head_dim").expect("pool_v is templatable");
    let fp8_prefill_k = fp8_kv_variant("paged_flash_prefill_hd256", kernels::PAGED_FLASH_PREFILL_HD256, "pool_k", "k_scales", "p.head_dim").expect("pool_k is templatable");
    let fp8_prefill = fp8_kv_variant(fp8_prefill_k.0, fp8_prefill_k.1, "pool_v", "v_scales", "p.head_dim").expect("pool_v is templatable");
    let fp8_scores = fp8_kv_variant("paged_decode_scores_batched", kernels::PAGED_DECODE_SCORES_BATCHED, "pool_k", "k_scales", "p.head_dim").expect("pool_k is templatable");
    let fp8_apply = fp8_kv_variant("paged_decode_apply_batched", kernels::PAGED_DECODE_APPLY_BATCHED, "pool_v", "v_scales", "p.head_dim").expect("pool_v is templatable");
    let mut decode = Vec::new();
    for (head_dim, group) in FLASH_SHAPES {
        let base = interned("paged_flash_decode_gqa_hd256", kernels::PAGED_FLASH_DECODE_GQA_HD256, &decode_params(head_dim, group)).expect("the decode kernel declares its constants");
        decode.push(base);
        decode.push(kv_tier_variant(base.0, base.1, CompactKv::Bf16).expect("the decode kernel marks its kv-tier blocks"));
        // The int8 and fp8 row loaders read 256-channel rows; at head_dim 128 those tiers keep the triad.
        if head_dim == 256 {
            decode.push(kv_tier_variant(base.0, base.1, CompactKv::Int8).expect("the decode kernel marks its kv-tier blocks"));
            decode.push(kv_tier_variant(base.0, base.1, CompactKv::Fp8).expect("the decode kernel marks its kv-tier blocks"));
        }
    }
    let all: Vec<(&'static str, &'static str)> = vec![
        (FLASH_COMBINE, kernels::PAGED_FLASH_DECODE_GQA_COMBINE),
        (F32_NAMES.append, kernels::PAGED_KV_APPEND_BATCHED),
        (F32_NAMES.scores, kernels::PAGED_DECODE_SCORES_BATCHED),
        (F32_NAMES.apply, kernels::PAGED_DECODE_APPLY_BATCHED),
        (F32_NAMES.prefill_hd256, kernels::PAGED_FLASH_PREFILL_HD256),
        dtype_variant_store("paged_kv_append_batched_word", kernels::PAGED_KV_APPEND_BATCHED_WORD, "pool", Dtype::BF16).expect("pool is a templatable store"),
        dtype_variant("paged_decode_scores_batched", kernels::PAGED_DECODE_SCORES_BATCHED, "pool_k", Dtype::BF16).expect("pool_k is templatable"),
        dtype_variant("paged_decode_apply_batched", kernels::PAGED_DECODE_APPLY_BATCHED, "pool_v", Dtype::BF16).expect("pool_v is templatable"),
        bf16_prefill,
        (INT8_NAMES.append, kernels::PAGED_KV_APPEND_I8_CLIPPED_BATCHED),
        (INT8_NAMES.scores, kernels::PAGED_DECODE_SCORES_I8_BATCHED),
        (INT8_NAMES.apply, kernels::PAGED_DECODE_APPLY_I8_BATCHED),
        i8_prefill,
        (FP8_NAMES.append, kernels::PAGED_KV_APPEND_FP8_CLIPPED_BATCHED),
        fp8_scores,
        fp8_apply,
        fp8_prefill,
    ];
    all.into_iter().chain(decode).filter(|(n, _)| !have.contains(n)).collect()
}

/// The fp8 tier for an engine that keeps its own int8 paged pools (`head_dim <= 128`, one scale per
/// (token slot, kv head), K and V in separate per-layer buffers - `qwen3`'s serving engine) and
/// owns the pipeline order itself: [`fp8_pool_pipelines`] is what it registers, and
/// [`Fp8PoolKernels::resolve`] finds the kernels by name. The pool layout is the int8 one byte for
/// byte, so the same buffers, sizes, offload and calibration clip tables serve both codecs; only
/// the kernels that write and read an element differ.
pub fn fp8_pool_pipelines() -> Vec<(&'static str, &'static str)> {
    use kernels::template::{fp8_decode_variant, fp8_kv_variant};
    vec![
        (FP8_NAMES.append, kernels::PAGED_KV_APPEND_FP8_CLIPPED_BATCHED),
        fp8_kv_variant("paged_decode_scores_batched", kernels::PAGED_DECODE_SCORES_BATCHED, "pool_k", "k_scales", "p.head_dim").expect("pool_k is templatable"),
        fp8_kv_variant("paged_decode_apply_batched", kernels::PAGED_DECODE_APPLY_BATCHED, "pool_v", "v_scales", "p.head_dim").expect("pool_v is templatable"),
        fp8_decode_variant("paged_flash_prefill_i8", kernels::PAGED_FLASH_PREFILL_I8).expect("the int8 prefill marks its decode block"),
        fp8_decode_variant("paged_flash_decode_split_i8", kernels::PAGED_FLASH_DECODE_SPLIT_I8).expect("the int8 split decode marks its decode block"),
    ]
}

/// Pipeline indices of [`fp8_pool_pipelines`] on one [`Gpu`].
#[derive(Clone, Copy, Debug)]
pub struct Fp8PoolKernels {
    /// Same bindings as `paged_kv_append_i8_clipped_batched`.
    pub append: usize,
    /// The f32 triad's scores kernel over an fp8 pool; the scales are bound LAST.
    pub scores: usize,
    /// The f32 triad's apply kernel over an fp8 pool; the scales are bound LAST.
    pub apply: usize,
    /// Same bindings and launch geometry as `paged_flash_prefill_i8`.
    pub prefill: usize,
    /// Same bindings and launch geometry as `paged_flash_decode_split_i8`.
    pub decode_split: usize,
}

impl Fp8PoolKernels {
    pub fn resolve(gpu: &Gpu) -> Result<Fp8PoolKernels, String> {
        let names: Vec<&str> = fp8_pool_pipelines().into_iter().map(|(n, _)| n).collect();
        let find = |i: usize| gpu.kernel_index(names[i]).ok_or_else(|| format!("fp8 KV: kernel '{}' is not registered on this Gpu (add model::kv_tier::fp8_pool_pipelines to its pipelines)", names[i]));
        Ok(Fp8PoolKernels { append: find(0)?, scores: find(1)?, apply: find(2)?, prefill: find(3)?, decode_split: find(4)? })
    }
}

/// The pipeline indices of one tier's kernels on one [`Gpu`], and the step
/// builders that bind a [`KvPlane`] to them.
pub struct KvKernels {
    tier: KvTier,
    append: usize,
    scores: usize,
    apply: usize,
    prefill_hd256: usize,
    /// Whether an fp32 plane's prefill may run the native fp16 tensor-core kernel
    /// where the device has one. On by default; [`Self::portable_prefill`] turns
    /// it off for a caller that needs the fp32 pipeline's own arithmetic (a test
    /// comparing tiers against it).
    native_prefill: bool,
    /// `(head_dim, group, pipeline)` of the fused decode, for the shapes this device was
    /// given a specialisation of; empty when the model did not register them
    /// (a missing fused decode is a slower decode, never a wrong one).
    decode: Vec<(u32, u32, usize)>,
    decode_combine: Option<usize>,
}

/// What a batched append writes: `batch` token rows (one per sequence, or one
/// per position of a prefill chunk), each to `(blocks[b], offsets[b])`.
#[derive(Clone, Copy, Debug)]
pub struct KvAppend {
    pub batch: u32,
    /// `n_kv_heads * head_dim`.
    pub kv_stride: u32,
    pub block_size: u32,
    pub head_dim: u32,
}

/// The fused `head_dim <= 256` prefill dispatch's shape: `n` query rows of ONE
/// sequence against its block table.
#[derive(Clone, Copy, Debug)]
pub struct PrefillShape {
    pub n: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    /// Rows per physical block.
    pub block_size: u32,
    /// Entries of each row's block table (`1` when one physical block backs the
    /// whole sequence).
    pub max_bt: u32,
}

/// The fused decode's shape: `batch` sequences, one query row each, against a
/// paged pool whose longest live sequence has `cap` keys.
#[derive(Clone, Copy, Debug)]
pub struct FlashDecodeShape {
    pub batch: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub block_size: u32,
    pub max_bt: u32,
    /// An upper bound on every `seq_lens` entry; it sizes the split count.
    pub cap: u32,
    /// Compute the split count for a batch of ONE sequence instead of this
    /// shape's `batch` - the exact-decode regime.
    ///
    /// The online-softmax grouping of a row's keys must not move when another
    /// sequence joins its step: with the splits pinned and `cap` held to the
    /// configured context, every row sees the same tile boundaries and the same
    /// number of partial records however many sequences share the batch, so its
    /// attention result is its own. A split past a sequence's live length costs
    /// one degenerate record, which is why sizing by the context is cheap
    /// (see [`flash_splits_for`]).
    pub pin_splits: bool,
}

/// Keys per tile of the fused decode kernel (`paged_flash_decode_gqa_hd256`'s
/// `TILE`) and tiles per split: a split covers 1024 keys, so a 128k context is
/// 128 splits per (sequence, kv head) and even one sequence fills the device.
const FLASH_TILE: u32 = 128;
const FLASH_TILES_PER_SPLIT: u32 = 8;
/// Workgroups a decode dispatch aims for: a card's SMs several times over, so a
/// short context is not left to a handful of workgroups each walking its keys
/// serially (at 1k tokens one sequence's two kv heads made FOUR workgroups and a
/// 346 us layer on a GH200, for 4 MB of keys and values).
const FLASH_TARGET_WORKGROUPS: u32 = 128;

/// `(splits, tiles per split)` for a decode of `batch` sequences over `n_kv_heads`
/// heads with room for `cap` keys: splits of up to [`FLASH_TILES_PER_SPLIT`] tiles,
/// cut finer - down to one tile - until the dispatch reaches
/// [`FLASH_TARGET_WORKGROUPS`]. A split past a sequence's live length costs one
/// degenerate partial record, so over-splitting a short sequence is cheap.
fn flash_splits(cap: u32, batch: u32, n_kv_heads: u32) -> (u32, u32) {
    flash_splits_for(FLASH_TARGET_WORKGROUPS, cap, batch, n_kv_heads, false)
}

/// [`flash_splits`] with the exact-decode pin: the count a batch of ONE
/// sequence gets, whatever `batch` is (`FlashDecodeShape::pin_splits`).
fn flash_splits_pinned(cap: u32, n_kv_heads: u32) -> (u32, u32) {
    flash_splits_for(FLASH_TARGET_WORKGROUPS, cap, 1, n_kv_heads, true)
}

/// Blocks per multiprocessor a native decode dispatch aims for: the native
/// kernel keeps several blocks resident per SM, and a decode read is latency
/// bound until each SM has enough rows in flight, so its grid is cut finer than
/// the WGSL tier's (which measured no gain from it: knowledge 350).
const FLASH_NATIVE_BLOCKS_PER_SM: u32 = 2;

/// [`flash_splits`] towards `target` workgroups. `pin_one` computes the count
/// for a batch of ONE sequence: the exact-decode regime's split, which must not
/// move when another sequence shares the step.
fn flash_splits_for(target: u32, cap: u32, batch: u32, n_kv_heads: u32, pin_one: bool) -> (u32, u32) {
    let tiles = cap.max(1).div_ceil(FLASH_TILE);
    let groups = (if pin_one { 1 } else { batch } * n_kv_heads).max(1);
    let wanted = target.div_ceil(groups).max(1);
    let per_split = tiles.div_ceil(wanted).clamp(1, FLASH_TILES_PER_SPLIT);
    (tiles.div_ceil(per_split), per_split)
}

/// Elements one thread of `paged_kv_append_batched_word` stores (mirrors the
/// kernel's `CHUNK`; even, so a packed bf16 word never straddles two threads).
const APPEND_WORD_CHUNK: u32 = 16;

/// Threads a `paged_kv_append_batched_word` dispatch must launch for `batch`
/// tokens of `kv_stride` elements.
pub fn append_word_threads(batch: u32, kv_stride: u32) -> u32 {
    batch * kv_stride.div_ceil(APPEND_WORD_CHUNK)
}

impl KvKernels {
    /// Look `tier`'s kernels up on `gpu` by name. An error names the first
    /// missing kernel: the model did not register [`kernel_list`].
    pub fn resolve(gpu: &Gpu, tier: KvTier) -> Result<KvKernels, String> {
        let names = tier.names();
        let find = |name: &str| gpu.kernel_index(name).ok_or_else(|| format!("KV tier {tier}: kernel '{name}' is not registered on this Gpu (add model::kv_tier::kernel_list to its pipelines)"));
        Ok(KvKernels {
            tier,
            append: find(names.append)?,
            scores: find(names.scores)?,
            apply: find(names.apply)?,
            prefill_hd256: find(names.prefill_hd256)?,
            native_prefill: true,
            decode: FLASH_SHAPES.iter().filter_map(|&(hd, group)| gpu.kernel_index(&decode_name(tier, hd, group)).map(|i| (hd, group, i))).collect(),
            decode_combine: gpu.kernel_index(FLASH_COMBINE),
        })
    }

    pub fn tier(&self) -> KvTier {
        self.tier
    }

    /// These kernels with the native fp16 prefill switched off: every tier then
    /// runs its registered pipeline, the fp32 one in fp32.
    pub fn portable_prefill(mut self) -> KvKernels {
        self.native_prefill = false;
        self
    }

    fn check(&self, plane: &KvPlane) {
        assert_eq!(plane.tier, self.tier, "KvKernels for {} bound to a {} plane", self.tier, plane.tier);
    }

    /// Write `src` (`[batch, kv_stride]` f32) into `plane` at each row's
    /// `(blocks[b], offsets[b])`. Overwrites, so a recycled slot carries
    /// nothing stale.
    pub fn append(&self, g: &Gpu, plane: &KvPlane, src: &DeviceBuffer, blocks: &DeviceBuffer, offsets: &DeviceBuffer, a: KvAppend) -> Step {
        self.check(plane);
        let KvAppend { batch, kv_stride, block_size, head_dim } = a;
        match self.tier.element {
            KvElement::F32 => g.step(self.append, &[src, blocks, offsets, &plane.data], &[batch, kv_stride, block_size], batch * kv_stride),
            // One thread per even-sized chunk of a token: a packed word is then private to one thread.
            KvElement::Bf16 => g.step(self.append, &[src, blocks, offsets, &plane.data], &[batch, kv_stride, block_size], append_word_threads(batch, kv_stride)),
            KvElement::Int8 | KvElement::Fp8E4m3 => {
                let (clip, scales) = (plane.clip.as_ref().expect("a scaled plane has a clip"), plane.scales.as_ref().expect("a scaled plane has scales"));
                g.step(self.append, &[src, blocks, offsets, clip, &plane.data, scales], &[batch, kv_stride, block_size, head_dim], batch * (kv_stride / head_dim))
            }
        }
    }

    /// `scores[b, h, j] = scale * q[b, h, :] . K[j]` for `j < seq_lens[b]`.
    pub fn scores(&self, g: &Gpu, q: &DeviceBuffer, k: &KvPlane, block_tables: &DeviceBuffer, seq_lens: &DeviceBuffer, scores: &DeviceBuffer, s: PagedDecodeShape) -> Step {
        self.check(k);
        let params = [s.batch, s.n_heads, s.group, s.head_dim, s.block_size, s.kv_stride, s.cap, s.max_bt, s.scale.to_bits()];
        let threads = s.batch * s.n_heads * s.cap;
        match (&k.scales, self.tier.element) {
            (None, _) => g.step(self.scores, &[q, &k.data, block_tables, seq_lens, scores], &params, threads),
            // The int8 kernel is written with its scales before the output; the fp8 one is a
            // template rewrite of the f32 kernel, which appends the scales binding last.
            (Some(sc), KvElement::Fp8E4m3) => g.step(self.scores, &[q, &k.data, block_tables, seq_lens, scores, sc], &params, threads),
            (Some(sc), _) => g.step(self.scores, &[q, &k.data, block_tables, seq_lens, sc, scores], &params, threads),
        }
    }

    /// `ctx[b, h, :] = sum_j probs[b, h, j] * V[j]` for `j < seq_lens[b]`.
    pub fn apply(&self, g: &Gpu, probs: &DeviceBuffer, v: &KvPlane, block_tables: &DeviceBuffer, seq_lens: &DeviceBuffer, ctx: &DeviceBuffer, s: PagedDecodeShape) -> Step {
        self.check(v);
        let params = [s.batch, s.n_heads, s.group, s.head_dim, s.block_size, s.kv_stride, s.cap, s.max_bt];
        let threads = s.batch * s.n_heads * s.head_dim;
        match (&v.scales, self.tier.element) {
            (None, _) => g.step(self.apply, &[probs, &v.data, block_tables, seq_lens, ctx], &params, threads),
            (Some(sc), KvElement::Fp8E4m3) => g.step(self.apply, &[probs, &v.data, block_tables, seq_lens, ctx, sc], &params, threads),
            (Some(sc), _) => g.step(self.apply, &[probs, &v.data, block_tables, seq_lens, sc, ctx], &params, threads),
        }
    }

    /// Whether [`Self::flash_decode`] can run this shape on `g`: a head width and GQA group the
    /// model registered a specialisation for ([`FLASH_SHAPES`]), and a device that runs
    /// workgroup-barrier kernels (not the CPU JIT, which keeps the scores/softmax/apply triad).
    pub fn flash_decode_available(&self, g: &Gpu, head_dim: u32, group: u32) -> bool {
        self.decode_combine.is_some() && self.decode.iter().any(|&(hd, gr, _)| hd == head_dim && gr == group) && g.caps().workgroup_reductions
    }

    /// Fused decode attention over `k`/`v` planes: attends each sequence's
    /// query row (`q`, `[batch, n_heads * 256]`) against its `seq_lens[b]`
    /// keys and writes `ctx` (`[batch, n_heads * 256]`). Two dispatches - the
    /// split-key pass and the merge - and no score slab. See
    /// `paged_flash_decode_gqa_hd256.wgsl` for the design and why it exists.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_decode(&self, g: &Gpu, q: &DeviceBuffer, k: &KvPlane, v: &KvPlane, block_tables: &DeviceBuffer, seq_lens: &DeviceBuffer, ctx: &DeviceBuffer, s: FlashDecodeShape) -> Vec<Step> {
        self.check(k);
        self.check(v);
        let group = s.n_heads / s.n_kv_heads;
        assert!(self.flash_decode_available(g, s.head_dim, group), "KvKernels::flash_decode: not available for head_dim {} group {group} on this device", s.head_dim);
        let decode = self.decode.iter().find(|&&(hd, gr, _)| hd == s.head_dim && gr == group).expect("checked by flash_decode_available").2;
        let (n_splits, tiles_per_split) = if s.pin_splits { flash_splits_pinned(s.cap, s.n_kv_heads) } else { flash_splits(s.cap, s.batch, s.n_kv_heads) };
        let mut params = [s.batch, s.n_heads, s.n_kv_heads, s.head_dim, group, s.block_size, s.max_bt, n_splits, tiles_per_split];
        // A native kernel's grid fills the device several blocks deep (the split count is a free choice: every split
        // writes its own record, and the combine merges any number of them).
        if g.native_kernel_for(decode, &params).is_some() {
            let target = g.caps().compute_units.map_or(FLASH_TARGET_WORKGROUPS, |cu| cu * FLASH_NATIVE_BLOCKS_PER_SM);
            (params[7], params[8]) = flash_splits_for(target, s.cap, s.batch, s.n_kv_heads, s.pin_splits);
        }
        let n_splits = params[7];
        // Every (sequence, head, split) record is written, a split with no live key included, so the slab needs no clear.
        let part = g.storage_uninit(s.batch as u64 * s.n_heads as u64 * n_splits as u64 * (s.head_dim as u64 + 2));
        let grid = Dispatch::Workgroups(s.batch * s.n_kv_heads * n_splits);
        let split = match (&k.scales, &v.scales) {
            (None, None) => g.dispatch(decode, &[q, &k.data, &v.data, block_tables, seq_lens, &part], &params, grid),
            (Some(ks), Some(vs)) => g.dispatch(decode, &[q, &k.data, &v.data, block_tables, seq_lens, &part, ks, vs], &params, grid),
            _ => unreachable!("check() pinned both planes to one tier"),
        };
        let combine = g.step(self.decode_combine.expect("checked by flash_decode_available"), &[&part, ctx], &[s.batch, s.n_heads, s.head_dim, n_splits], s.batch * s.n_heads * 256);
        vec![split, combine]
    }

    /// The native fp16 tensor-core prefill dispatch over `k`/`v` planes alone: `None` when the device, the tier, the head width or
    /// the round's row count has no such kernel (or [`Self::portable_prefill`] withheld it), and the caller keeps its own path.
    /// Same buffers and uniform as the portable kernel; it also serves head width 128, which the portable one does not. The
    /// round is ONE sequence (the tables' rows repeat it), which is what lets a compact head-256 cache take the warpgroup
    /// kernel and its merge of key splits: more than one step.
    pub fn native_flash_prefill(&self, g: &Gpu, q: &DeviceBuffer, k: &KvPlane, v: &KvPlane, block_ids: &DeviceBuffer, seq_lens: &DeviceBuffer, ctx: &DeviceBuffer, s: PrefillShape) -> Option<Vec<Step>> {
        self.check(k);
        self.check(v);
        if !self.native_prefill {
            return None;
        }
        let params = [s.n, s.n_heads, s.n_kv_heads, s.head_dim, s.n_heads / s.n_kv_heads, s.block_size, s.max_bt];
        let blocks = s.n_heads * s.n.div_ceil(64);
        match (&k.scales, &v.scales) {
            (None, None) => {
                // The fp32 and bf16 tiers' planes are what the fp16 tensor-core kernels read (same buffers, uniform and
                // geometry as the portable one).
                let bufs = [q, &k.data, &v.data, block_ids, seq_lens, ctx];
                match self.tier.element {
                    KvElement::F32 => crate::block::native_paged_flash_prefill(g, s.head_dim, &bufs, &params, blocks),
                    KvElement::Bf16 => crate::block::native_paged_flash_prefill_bf16(g, s.head_dim, &bufs, &params, blocks),
                    KvElement::Int8 | KvElement::Fp8E4m3 => None,
                }
                .map(|step| vec![step])
            }
            (Some(ks), Some(vs)) => {
                let bufs = [q, &k.data, &v.data, block_ids, seq_lens, ctx, ks, vs];
                let fp8 = self.tier.element == KvElement::Fp8E4m3;
                crate::block::native_paged_flash_prefill_wg(g, s.head_dim, &bufs, &params, fp8).or_else(|| {
                    if fp8 {
                        crate::block::native_paged_flash_prefill_kvfp8(g, s.head_dim, &bufs, &params, blocks)
                    } else {
                        crate::block::native_paged_flash_prefill_kv8(g, s.head_dim, &bufs, &params, blocks)
                    }
                    .map(|step| vec![step])
                })
            }
            _ => unreachable!("check() pinned both planes to one tier"),
        }
    }

    /// The fused causal prefill dispatch over `k`/`v` planes (one pipeline for
    /// every tier; the compact ones read their planes in place): the native kernel where there is one, else the portable
    /// `head_dim = 256` pipeline.
    pub fn flash_prefill_hd256(&self, g: &Gpu, q: &DeviceBuffer, k: &KvPlane, v: &KvPlane, block_ids: &DeviceBuffer, seq_lens: &DeviceBuffer, ctx: &DeviceBuffer, s: PrefillShape) -> Vec<Step> {
        if let Some(native) = self.native_flash_prefill(g, q, k, v, block_ids, seq_lens, ctx, s) {
            return native;
        }
        let params = [s.n, s.n_heads, s.n_kv_heads, s.head_dim, s.n_heads / s.n_kv_heads, s.block_size, s.max_bt];
        let grid = Dispatch::Workgroups(s.n_heads * s.n.div_ceil(64));
        vec![match (&k.scales, &v.scales) {
            (None, None) => g.dispatch(self.prefill_hd256, &[q, &k.data, &v.data, block_ids, seq_lens, ctx], &params, grid),
            (Some(ks), Some(vs)) => g.dispatch(self.prefill_hd256, &[q, &k.data, &v.data, block_ids, seq_lens, ctx, ks, vs], &params, grid),
            _ => unreachable!("check() pinned both planes to one tier"),
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Seeding one window of a plane from another copies the rows (and, in the int8
    /// tier, their scales) bit for bit, binds the plane ONCE (a buffer that is
    /// `read_write` and also another binding of one dispatch is a wgpu validation
    /// error on a GPU, and `Gpu::dispatch` refuses it on every backend), and leaves
    /// every other row alone.
    #[test]
    fn copying_rows_within_a_plane_is_exact_for_data_and_scales() {
        let gpu = Gpu::new_cpu(&[("copy_within", kernels::COPY_WITHIN)]);
        let copy_within = gpu.kernel_index("copy_within").unwrap();
        let (rows, kv_stride, head_dim) = (8u64, 8u64, 4u64);
        let plane = KvPlane::new(&gpu, KvTier::INT8, rows, kv_stride, head_dim);
        let scales = plane.scales.as_ref().expect("the int8 tier keeps per-row scales");
        let words = |base: u32, n: usize| (0..n as u32).map(|i| f32::from_bits(base + i * 0x101)).collect::<Vec<_>>();
        let (data_words, scale_words) = (rows * kv_stride / 4, rows * kv_stride / head_dim);
        gpu.write(plane.data(), &words(0x4000_0000, data_words as usize).iter().map(|f| f.to_bits()).collect::<Vec<_>>());
        gpu.write(scales, &words(0x3f80_0000, scale_words as usize).iter().map(|f| f.to_bits()).collect::<Vec<_>>());
        let before = |buf: &DeviceBuffer, n: u64| gpu.read(buf, n as usize).iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        let (data0, scales0) = (before(plane.data(), data_words), before(scales, scale_words));

        let steps = plane.copy_rows(&gpu, copy_within, 0, 4, 2, kv_stride, head_dim);
        gpu.submit(&[], &steps);
        let (data1, scales1) = (before(plane.data(), data_words), before(scales, scale_words));

        let (dw, sw) = ((kv_stride / 4) as usize, (kv_stride / head_dim) as usize);
        let mut want_data = data0.clone();
        want_data.copy_within(0..2 * dw, 4 * dw);
        let mut want_scales = scales0.clone();
        want_scales.copy_within(0..2 * sw, 4 * sw);
        assert_eq!(data1, want_data);
        assert_eq!(scales1, want_scales);
        assert_ne!(data1, data0, "the copy must have changed something");
    }

    #[test]
    #[should_panic(expected = "overlap")]
    fn an_overlapping_copy_is_refused() {
        let gpu = Gpu::new_cpu(&[("copy_within", kernels::COPY_WITHIN)]);
        let buf = gpu.storage(16);
        let _ = copy_words_within(&gpu, 0, &buf, 0, 4, 8);
    }

    /// A long context keeps its 1024-key splits (the device is full anyway); a
    /// short one is cut finer so even one sequence reaches the workgroup target,
    /// and the splits always cover the cap.
    #[test]
    fn short_contexts_are_split_finer_and_long_ones_keep_whole_splits() {
        assert_eq!(flash_splits(131_072, 1, 2), (128, 8));
        assert_eq!(flash_splits(1288, 1, 2), (11, 1), "one tile per split: 11 splits x 2 kv heads");
        assert_eq!(flash_splits(1288, 16, 2), (4, 3), "a batch of 16 already makes 32 groups: 4 splits each reach the target");
        assert_eq!(flash_splits(1, 1, 2), (1, 1));
        for (cap, batch, nkv) in [(1u32, 1u32, 1u32), (127, 3, 2), (128, 3, 2), (129, 64, 2), (4096, 1, 2), (50_000, 7, 4), (1_000_000, 1, 2)] {
            let (n, per) = flash_splits(cap, batch, nkv);
            assert!((1..=FLASH_TILES_PER_SPLIT).contains(&per));
            assert!(n * per * FLASH_TILE >= cap, "splits cover the cap");
            assert!((n - 1) * per * FLASH_TILE < cap.max(1), "no wholly empty trailing split");
        }
    }

    /// The exact-decode regime pins the split count to the batch-1 value, so a
    /// row's key partition is a function of `cap` and `n_kv_heads` alone. The
    /// unpinned count is batch-dependent by design (it aims the dispatch at the
    /// workgroup target), which is exactly what a row must not see.
    #[test]
    fn pinned_splits_are_the_single_sequence_splits_at_every_batch() {
        for (cap, nkv) in [(1u32, 1u32), (127, 2), (1288, 2), (4096, 4), (8192, 2), (32_768, 4), (131_072, 2)] {
            for target in [FLASH_TARGET_WORKGROUPS, FLASH_TARGET_WORKGROUPS * 8] {
                let one = flash_splits_for(target, cap, 1, nkv, true);
                for batch in 1..=8 {
                    assert_eq!(flash_splits_for(target, cap, batch, nkv, true), one, "cap {cap} batch {batch} nkv {nkv} target {target}");
                }
            }
            assert_eq!(flash_splits_pinned(cap, nkv), flash_splits(cap, 1, nkv));
        }
        // The pinned answer is not vacuous: at this shape the unpinned count
        // moves with the batch, which is the co-batch dependence being removed.
        assert_ne!(flash_splits(8192, 8, 2), flash_splits(8192, 1, 2));
    }

    #[test]
    fn tiers_parse_by_their_printed_names_and_refuse_the_rest() {
        for t in KvTier::ALL {
            assert_eq!(KvTier::parse(t.as_str()), Ok(t));
        }
        assert_eq!(KvTier::parse("FP32"), Ok(KvTier::F32));
        assert_eq!(KvTier::parse(" i8 "), Ok(KvTier::INT8));
        assert_eq!(KvTier::parse("E4M3"), Ok(KvTier::FP8));
        assert!(KvTier::parse("fp4").unwrap_err().contains("f32, bf16, int8 or fp8"));
    }

    /// The numbers a planner quotes: Qwen3.8-27B's GQA layer is
    /// `kv_stride = 4 * 256` and a 128k sequence is 131072 rows.
    #[test]
    fn a_planes_bytes_halve_for_bf16_and_quarter_for_int8_plus_its_scales() {
        let (rows, stride, hd) = (131_072u64, 1024u64, 256u64);
        let f32b = KvTier::F32.plane_bytes(rows, stride, hd);
        assert_eq!(f32b, rows * stride * 4);
        assert_eq!(KvTier::BF16.plane_bytes(rows, stride, hd), f32b / 2);
        // 1 byte per element plus one f32 scale per (row, kv head).
        assert_eq!(KvTier::INT8.plane_bytes(rows, stride, hd), rows * stride + rows * 4 * 4);
        assert!(KvTier::INT8.plane_bytes(rows, stride, hd) * 3 < f32b, "int8 must be more than 3x smaller");
    }

    /// FP8 E4M3 stores one byte per element and one f32 scale per row, exactly the int8
    /// footprint, so a planner may swap the two without re-sizing a pool.
    #[test]
    fn an_fp8_plane_has_the_int8_footprint() {
        for (rows, stride, hd) in [(131_072u64, 1024u64, 256u64), (4096, 1024, 128), (7, 8, 4)] {
            assert_eq!(KvTier::FP8.plane_words(rows, stride, hd), KvTier::INT8.plane_words(rows, stride, hd));
        }
    }

    /// The host reference decode names the format's landmarks: the largest finite value 448,
    /// the smallest normal 2^-6, the smallest subnormal 2^-9, and signed zero.
    #[test]
    fn the_e4m3_reference_decode_hits_the_formats_landmarks() {
        assert_eq!(decode_e4m3(0x7E), 448.0);
        assert_eq!(decode_e4m3(0xFE), -448.0);
        assert_eq!(decode_e4m3(0x08), 2f32.powi(-6));
        assert_eq!(decode_e4m3(0x01), 2f32.powi(-9));
        assert_eq!(decode_e4m3(0x38), 1.0);
        assert_eq!(decode_e4m3(0x00), 0.0);
    }

    /// The descriptor is the checkpoint crate's encoding: every tier round-trips through
    /// it, its bytes are the encoding's bytes, and an encoding the kernels cannot decode
    /// is named in the refusal.
    #[test]
    fn a_tier_is_the_checkpoint_encoding_of_its_bytes() {
        for t in KvTier::ALL {
            assert_eq!(KvTier::from_encoding(&t.encoding()), Ok(t));
            let d = t.plane_descriptor(4096, 1024, 128);
            d.validate().unwrap();
            assert_eq!(t.plane_bytes(4096, 1024, 128), d.payload_bytes() + if d.encoding.scales.is_empty() { 0 } else { d.scale_bytes(0) });
        }
        let err = KvTier::from_encoding(&QuantEncoding::nf4_blocked(32)).unwrap_err();
        assert!(err.contains("no KV tier"), "{err}");
        assert!(KvTier::from_encoding(&QuantEncoding::fp8_e4m3_tiled(128)).is_err(), "tile-scaled fp8 is a weight layout, not a KV one");
    }

    #[test]
    fn an_odd_element_count_still_gets_a_whole_trailing_word() {
        assert_eq!(KvTier::BF16.plane_words(1, 3, 3).data, 2);
        assert_eq!(KvTier::INT8.plane_words(1, 5, 5).data, 2);
    }

    #[test]
    fn addressing_stops_at_the_u32_element_index() {
        assert!(KvTier::fits_addressing(4 * 1024 * 1024, 1024), "2^32 elements: the last index is 2^32 - 1");
        assert!(!KvTier::fits_addressing(4 * 1024 * 1024 + 1, 1024));
    }

    #[test]
    fn an_unset_environment_variable_is_f32() {
        assert_eq!(KvTier::from_env("KV_TIER_TEST_UNSET_VARIABLE"), Ok(KvTier::F32));
    }
}

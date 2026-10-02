// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Compact KV-cache tiers for batched paged decode: how a GQA layer's K and V
//! planes are stored (`f32`, `bf16` or per-row `int8`), and the kernels that
//! append to, and attend over, a plane of each tier.
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
//!
//! The kernels are the repo's existing paged-KV family - nothing here is a
//! second implementation of attention:
//!
//! * `bf16`: `paged_kv_append_batched_word` (packing store) and the
//!   `paged_decode_{scores,apply}_batched` loads, both rewritten by
//!   [`kernels::template`]'s bf16 variants;
//! * `int8`: `paged_kv_append_i8_clipped_batched`, `paged_decode_scores_i8_batched`
//!   and `paged_decode_apply_i8_batched`;
//! * the fused `head_dim = 256` prefill kernel, whose bf16 and int8 loads are
//!   template variants of the one `f32` source.
//!
//! [`kernel_list`] is what a model registers on its [`Gpu`] to run any tier;
//! [`KvKernels::resolve`] looks the chosen tier's kernels up BY NAME.
//!
//! # Addressing limit
//!
//! The kernels index a pool with `u32` element offsets, so one plane may hold
//! at most [`MAX_PLANE_ELEMS`] elements ([`KvTier::fits_addressing`]). At
//! `kv_stride = 1024` that is 4 194 303 token rows - 32 sequences of 128k.

use gpu_core::{DeviceBuffer, Dispatch, Gpu, Step};

use crate::ops::PagedDecodeShape;

/// Largest number of elements one plane may hold: the kernels' `u32` element
/// index. Past it a pool would silently alias its own start.
pub const MAX_PLANE_ELEMS: u64 = u32::MAX as u64;

/// How a KV plane is stored. See the module doc for the layouts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KvTier {
    F32,
    Bf16,
    Int8,
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
    /// Every tier, in order of decreasing size.
    pub const ALL: [KvTier; 3] = [KvTier::F32, KvTier::Bf16, KvTier::Int8];

    /// Parse a user-facing name: `f32`/`fp32`, `bf16`, `int8`/`i8`.
    pub fn parse(s: &str) -> Result<KvTier, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f32" | "fp32" => Ok(KvTier::F32),
            "bf16" => Ok(KvTier::Bf16),
            "int8" | "i8" => Ok(KvTier::Int8),
            other => Err(format!("unknown KV tier {other:?}: expected f32, bf16 or int8")),
        }
    }

    /// The name [`Self::parse`] accepts, and the one reports print.
    pub fn as_str(self) -> &'static str {
        match self {
            KvTier::F32 => "f32",
            KvTier::Bf16 => "bf16",
            KvTier::Int8 => "int8",
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
        let elems = rows * kv_stride;
        match self {
            KvTier::F32 => PlaneWords { data: elems, scales: 0 },
            KvTier::Bf16 => PlaneWords { data: elems.div_ceil(2), scales: 0 },
            KvTier::Int8 => PlaneWords { data: elems.div_ceil(4), scales: rows * (kv_stride / head_dim) },
        }
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
}

impl KvPlane {
    /// A plane of `rows` token rows of `kv_stride = n_kv_heads * head_dim`
    /// elements. Panics if the plane cannot be addressed ([`MAX_PLANE_ELEMS`])
    /// or `kv_stride` is not a whole number of heads, or - int8 - the head is
    /// not a whole number of packed words.
    pub fn new(gpu: &Gpu, tier: KvTier, rows: u64, kv_stride: u64, head_dim: u64) -> KvPlane {
        assert!(head_dim > 0 && kv_stride.is_multiple_of(head_dim), "KvPlane: kv_stride {kv_stride} is not a whole number of {head_dim}-wide heads");
        assert!(KvTier::fits_addressing(rows, kv_stride), "KvPlane: {rows} rows x {kv_stride} elements exceed the kernels' u32 element index ({MAX_PLANE_ELEMS})");
        let words = tier.plane_words(rows, kv_stride, head_dim);
        match tier {
            KvTier::F32 | KvTier::Bf16 => KvPlane { tier, data: gpu.storage(words.data), scales: None, clip: None },
            KvTier::Int8 => {
                assert!(head_dim.is_multiple_of(4), "KvPlane: an int8 head of {head_dim} elements is not a whole number of packed words");
                let n_kv = (kv_stride / head_dim) as usize;
                KvPlane {
                    tier,
                    data: gpu.storage(words.data),
                    scales: Some(gpu.storage(words.scales)),
                    clip: Some(gpu.storage_init("kv.clip", &vec![f32::MAX; n_kv])),
                }
            }
        }
    }

    /// A one-word `f32` stand-in for a layer that keeps no KV (a GDN layer, or
    /// one another shard owns): every layer index carries a plane, so callers
    /// never branch on layer type to index.
    pub fn placeholder(gpu: &Gpu) -> KvPlane {
        KvPlane { tier: KvTier::F32, data: gpu.storage(1), scales: None, clip: None }
    }

    pub fn tier(&self) -> KvTier {
        self.tier
    }

    /// The plane's first `rows` rows decoded to `f32` on the host, `[rows,
    /// kv_stride]`: what an attention kernel sees, for diagnostics and for
    /// tests that compare a tier's loads against `f32` ones over the same
    /// values. Reads the whole plane back - never on a serving path.
    pub fn read_dequantized(&self, gpu: &Gpu, rows: usize, kv_stride: usize, head_dim: usize) -> Vec<f32> {
        let elems = rows * kv_stride;
        match self.tier {
            KvTier::F32 => gpu.read(&self.data, elems),
            KvTier::Bf16 => {
                let words = gpu.read(&self.data, elems.div_ceil(2));
                (0..elems).map(|i| f32::from_bits(((words[i / 2].to_bits() >> (16 * (i % 2))) & 0xFFFF) << 16)).collect()
            }
            KvTier::Int8 => {
                let words = gpu.read(&self.data, elems.div_ceil(4));
                let n_kv = kv_stride / head_dim;
                let scales = gpu.read(self.scales.as_ref().expect("int8 plane has scales"), rows * n_kv);
                (0..elems)
                    .map(|i| {
                        let byte = (words[i / 4].to_bits() >> (8 * (i % 4))) & 0xFF;
                        f32::from(byte as u8 as i8) * scales[i / head_dim]
                    })
                    .collect()
            }
        }
    }

    /// The element storage, for kernels outside this module that read an `f32`
    /// plane directly.
    pub fn data(&self) -> &DeviceBuffer {
        &self.data
    }
}

/// One layer's K and V planes.
pub struct KvLayer {
    pub k: KvPlane,
    pub v: KvPlane,
}

impl KvLayer {
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

impl KvTier {
    fn names(self) -> &'static Names {
        match self {
            KvTier::F32 => &F32_NAMES,
            KvTier::Bf16 => &BF16_NAMES,
            KvTier::Int8 => &INT8_NAMES,
        }
    }
}

/// Every `(name, wgsl source)` pair [`KvKernels::resolve`] can ask a [`Gpu`]
/// for, across all tiers - what a model that offers the compact tiers adds to
/// its own pipeline list. Names already in `existing` are not repeated.
pub fn kernel_list(existing: &[(&str, &str)]) -> Vec<(&'static str, &'static str)> {
    use kernels::template::{dtype_variant, dtype_variant_store, int8_kv_variant};
    use gpu_core::select::Dtype;

    let have: std::collections::HashSet<&str> = existing.iter().map(|(n, _)| *n).collect();
    let bf16_prefill_k = dtype_variant("paged_flash_prefill_hd256", kernels::PAGED_FLASH_PREFILL_HD256, "pool_k", Dtype::BF16).expect("pool_k is templatable");
    let bf16_prefill = dtype_variant(bf16_prefill_k.0, bf16_prefill_k.1, "pool_v", Dtype::BF16).expect("pool_v is templatable");
    let i8_prefill_k = int8_kv_variant("paged_flash_prefill_hd256", kernels::PAGED_FLASH_PREFILL_HD256, "pool_k", "k_scales", "p.head_dim").expect("pool_k is templatable");
    let i8_prefill = int8_kv_variant(i8_prefill_k.0, i8_prefill_k.1, "pool_v", "v_scales", "p.head_dim").expect("pool_v is templatable");
    let all: Vec<(&'static str, &'static str)> = vec![
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
    ];
    all.into_iter().filter(|(n, _)| !have.contains(n)).collect()
}

/// The pipeline indices of one tier's kernels on one [`Gpu`], and the step
/// builders that bind a [`KvPlane`] to them.
pub struct KvKernels {
    tier: KvTier,
    append: usize,
    scores: usize,
    apply: usize,
    prefill_hd256: usize,
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
/// sequence against a single-block pool window.
#[derive(Clone, Copy, Debug)]
pub struct PrefillShape {
    pub n: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    /// Rows per physical block (the per-sequence capacity, `max_bt = 1`).
    pub block_size: u32,
}

impl KvKernels {
    /// Look `tier`'s kernels up on `gpu` by name. An error names the first
    /// missing kernel: the model did not register [`kernel_list`].
    pub fn resolve(gpu: &Gpu, tier: KvTier) -> Result<KvKernels, String> {
        let names = tier.names();
        let find = |name: &str| gpu.kernel_index(name).ok_or_else(|| format!("KV tier {tier}: kernel '{name}' is not registered on this Gpu (add model::kv_tier::kernel_list to its pipelines)"));
        Ok(KvKernels { tier, append: find(names.append)?, scores: find(names.scores)?, apply: find(names.apply)?, prefill_hd256: find(names.prefill_hd256)? })
    }

    pub fn tier(&self) -> KvTier {
        self.tier
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
        match self.tier {
            KvTier::F32 => g.step(self.append, &[src, blocks, offsets, &plane.data], &[batch, kv_stride, block_size], batch * kv_stride),
            // One thread per token: a packed word is then private to one thread.
            KvTier::Bf16 => g.step(self.append, &[src, blocks, offsets, &plane.data], &[batch, kv_stride, block_size], batch),
            KvTier::Int8 => {
                let (clip, scales) = (plane.clip.as_ref().expect("int8 plane has a clip"), plane.scales.as_ref().expect("int8 plane has scales"));
                g.step(self.append, &[src, blocks, offsets, clip, &plane.data, scales], &[batch, kv_stride, block_size, head_dim], batch * (kv_stride / head_dim))
            }
        }
    }

    /// `scores[b, h, j] = scale * q[b, h, :] . K[j]` for `j < seq_lens[b]`.
    pub fn scores(&self, g: &Gpu, q: &DeviceBuffer, k: &KvPlane, block_tables: &DeviceBuffer, seq_lens: &DeviceBuffer, scores: &DeviceBuffer, s: PagedDecodeShape) -> Step {
        self.check(k);
        let params = [s.batch, s.n_heads, s.group, s.head_dim, s.block_size, s.kv_stride, s.cap, s.max_bt, s.scale.to_bits()];
        let threads = s.batch * s.n_heads * s.cap;
        match &k.scales {
            None => g.step(self.scores, &[q, &k.data, block_tables, seq_lens, scores], &params, threads),
            Some(sc) => g.step(self.scores, &[q, &k.data, block_tables, seq_lens, sc, scores], &params, threads),
        }
    }

    /// `ctx[b, h, :] = sum_j probs[b, h, j] * V[j]` for `j < seq_lens[b]`.
    pub fn apply(&self, g: &Gpu, probs: &DeviceBuffer, v: &KvPlane, block_tables: &DeviceBuffer, seq_lens: &DeviceBuffer, ctx: &DeviceBuffer, s: PagedDecodeShape) -> Step {
        self.check(v);
        let params = [s.batch, s.n_heads, s.group, s.head_dim, s.block_size, s.kv_stride, s.cap, s.max_bt];
        let threads = s.batch * s.n_heads * s.head_dim;
        match &v.scales {
            None => g.step(self.apply, &[probs, &v.data, block_tables, seq_lens, ctx], &params, threads),
            Some(sc) => g.step(self.apply, &[probs, &v.data, block_tables, seq_lens, sc, ctx], &params, threads),
        }
    }

    /// The fused causal prefill dispatch over `k`/`v` planes (one pipeline for
    /// every tier; the compact ones read their planes in place).
    #[allow(clippy::too_many_arguments)]
    pub fn flash_prefill_hd256(&self, g: &Gpu, q: &DeviceBuffer, k: &KvPlane, v: &KvPlane, block_ids: &DeviceBuffer, seq_lens: &DeviceBuffer, ctx: &DeviceBuffer, s: PrefillShape) -> Step {
        self.check(k);
        self.check(v);
        let params = [s.n, s.n_heads, s.n_kv_heads, s.head_dim, s.n_heads / s.n_kv_heads, s.block_size, 1];
        let grid = Dispatch::Workgroups(s.n_heads * s.n.div_ceil(64));
        match (&k.scales, &v.scales) {
            (None, None) => g.dispatch(self.prefill_hd256, &[q, &k.data, &v.data, block_ids, seq_lens, ctx], &params, grid),
            (Some(ks), Some(vs)) => g.dispatch(self.prefill_hd256, &[q, &k.data, &v.data, block_ids, seq_lens, ctx, ks, vs], &params, grid),
            _ => unreachable!("check() pinned both planes to one tier"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiers_parse_by_their_printed_names_and_refuse_the_rest() {
        for t in KvTier::ALL {
            assert_eq!(KvTier::parse(t.as_str()), Ok(t));
        }
        assert_eq!(KvTier::parse("FP32"), Ok(KvTier::F32));
        assert_eq!(KvTier::parse(" i8 "), Ok(KvTier::Int8));
        assert!(KvTier::parse("fp8").unwrap_err().contains("f32, bf16 or int8"));
    }

    /// The numbers a planner quotes: Qwen3.8-27B's GQA layer is
    /// `kv_stride = 4 * 256` and a 128k sequence is 131072 rows.
    #[test]
    fn a_planes_bytes_halve_for_bf16_and_quarter_for_int8_plus_its_scales() {
        let (rows, stride, hd) = (131_072u64, 1024u64, 256u64);
        let f32b = KvTier::F32.plane_bytes(rows, stride, hd);
        assert_eq!(f32b, rows * stride * 4);
        assert_eq!(KvTier::Bf16.plane_bytes(rows, stride, hd), f32b / 2);
        // 1 byte per element plus one f32 scale per (row, kv head).
        assert_eq!(KvTier::Int8.plane_bytes(rows, stride, hd), rows * stride + rows * 4 * 4);
        assert!(KvTier::Int8.plane_bytes(rows, stride, hd) * 3 < f32b, "int8 must be more than 3x smaller");
    }

    #[test]
    fn an_odd_element_count_still_gets_a_whole_trailing_word() {
        assert_eq!(KvTier::Bf16.plane_words(1, 3, 3).data, 2);
        assert_eq!(KvTier::Int8.plane_words(1, 5, 5).data, 2);
    }

    #[test]
    fn addressing_stops_at_the_u32_element_index() {
        assert!(KvTier::fits_addressing(4 * 1024 * 1024 - 1, 1024));
        assert!(!KvTier::fits_addressing(4 * 1024 * 1024, 1024));
    }

    #[test]
    fn an_unset_environment_variable_is_f32() {
        assert_eq!(KvTier::from_env("BRAIN_KV_TIER_TEST_UNSET_VARIABLE"), Ok(KvTier::F32));
    }
}

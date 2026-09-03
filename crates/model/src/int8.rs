// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared int8 (DP4A) weight quantization — the host half of the engine's
//! int8 inference tier.
//!
//! One implementation for every model that runs the DP4A path (zimage DiT,
//! Qwen encoder/decoder, FLUX.2 DiT): weights are quantized ONCE at build with
//! [`quantize_weight`]; activations are quantized on-device each forward with a
//! dynamic per-token scale (`max_abs_row` → `quant_pack`), then the DP4A GEMM
//! (`matmul_i8_dyn`, ~4× the fp32 rate on Pascal) dequantizes with `sx·sw`.
//! The packed layout here is exactly what `matmul_i8*.wgsl` consume — if it
//! changes, it changes for every model at once, which is the point.
//!
//! ## Why the weight scale is GROUP-wise, not per output channel
//!
//! The scale granularity along the reduction axis `k` is [`GROUP`] = 32
//! elements, matching GGUF's `Q8_0` block exactly. It is not a tunable: a
//! scale spanning a whole output channel lets ONE outlier weight set the
//! quantization step for every other weight in that channel, and `k` reaches
//! 17408 in the models this tier serves. That is a measured defect in this
//! repo, not a theoretical one - whole-channel INT8 on the Qwen3 KV-cache
//! decode graph measured cosine 0.994 / max_abs 11.2 against fp32's 1.000000
//! / 0.0002 (`AGENTS.md`, "Weight-only quantization must be block/group-wise";
//! the NPU-side quantizer `npu::topo::linear_quant` has used `QUANT_GROUP =
//! 32` for the same reason since that measurement).
//!
//! Matching `Q8_0`'s 32 exactly is also what keeps the door open for a
//! byte-level repack: a `Q8_0` block is one f16 `d` plus 32 int8, so a raw
//! `Q8_0` tensor can become this layout by widening `d` to f32 and re-packing
//! the same 32 quantized bytes four-to-a-word, with **no** requantization and
//! therefore no further precision loss.

/// Weight-scale group size along the reduction axis `k`, in int8 elements -
/// `Q8_0`'s block size. Eight packed `u32` words (4 int8 each) per group, so a
/// group boundary is always a word boundary and every kernel's k-chunk of 8
/// words is exactly one group.
pub const GROUP: usize = 32;

/// Packed `u32` words per scale group (`GROUP / 4`).
pub const WORDS_PER_GROUP: usize = GROUP / 4;

/// Elements in the scale tensor of an `[n, k]` int8 weight: `n * k/32`, i.e.
/// the scale is `[n, k/GROUP]` row-major, the same row order as the packed
/// words. THE definition - every producer and consumer sizes against this
/// rather than open-coding the division.
#[inline]
pub const fn scale_len(n: usize, k: usize) -> usize {
    n * (k / GROUP)
}

/// Reject a scale tensor whose length is not [`scale_len`] - and say so in the
/// words a reader of an OLD checkpoint needs, because the one length that
/// arrives here by accident is `n`: the whole-channel layout brain wrote
/// before the group-wise one. Silently reading `[n]` as `[n, k/32]` would run
/// and produce plausible garbage, so every loader of a packed int8 weight
/// funnels its scale sibling through this.
pub fn check_scale_len(name: &str, got: usize, n: usize, k: usize) -> Result<(), String> {
    let want = scale_len(n, k);
    if got == want {
        return Ok(());
    }
    let hint = if got == n {
        " - that is the WHOLE-CHANNEL ([n]) int8 scale layout brain no longer reads; the on-disk format changed to group-wise (GROUP=32) and this checkpoint must be re-imported"
    } else {
        ""
    };
    Err(format!("int8 weight '{name}': scale has {got} entries, expected {want} for [n={n}, k/{GROUP}={}]{hint}", k / GROUP))
}

/// Group-wise symmetric int8 quantization of an `[n, k]` weight: ONE scale per
/// [`GROUP`]-element block of the reduction axis, packed into `[n, k/4]` u32
/// (4 int8 per u32, little-endian along K). Returns `(packed, scales)` with
/// `scales` shaped `[n, k/GROUP]` row-major and
/// `scales[r, g] = max|w[r, 32g .. 32g+32]| / 127`.
///
/// `k` must be a multiple of [`GROUP`] - asserted, never silently padded. That
/// is `Q8_0`'s own requirement and it subsumes the `k % 4 == 0` the u32
/// packing needs; a caller whose `k` does not divide by 32 must keep that
/// tensor in fp32 rather than quantize a ragged tail.
///
/// Row-parallel through `backend_cpu::par` (the workspace's only rayon seam,
/// the same one-pool-one-policy scheduler `model::parallel`/`model::shard`
/// already route their own host-parallel reductions through). Every output
/// row reads only its own `k` inputs and writes only its own `k/32` scales and
/// `k/4` words, so the split is a scheduling change and the result is
/// bit-identical to the serial form - pinned by `tests/quantize_weight_is_
/// schedule_invariant.rs` against a serial oracle, not asserted here. This
/// matters because quantizing a real checkpoint is not a build-time nicety on
/// every model: a streamed int8 tier re-quantizes hundreds of megabytes per
/// transformer block, where the serial loop was the single largest host-CPU
/// cost of a generation's first denoise step.
pub fn quantize_weight(w: &[f32], n: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
    assert_eq!(k % GROUP, 0, "int8 K must be a multiple of {GROUP} (got {k})");
    assert_eq!(w.len(), n * k, "weight len {} != n*k {}", w.len(), n * k);
    let kg = k / 4;
    let gs = k / GROUP;
    // ONE pass over `w`: a row's scales and its packed words are produced by
    // the same task, from the same read. The two outputs have different
    // element types AND different widths (`k/4` u32 against `k/32` f32),
    // which is what `chunks2_mut` exists for.
    let mut packed = vec![0u32; n * kg];
    let mut sw = vec![0f32; n * gs];
    backend_cpu::par::chunks2_mut(&mut packed, kg, &mut sw, gs, |r, prow, srow| {
        let row = &w[r * k..r * k + k];
        group_scales(row, srow);
        pack_row(row, srow, prow);
    });
    (packed, sw)
}

/// One row's `k/GROUP` scales, written into `out`: `max|block| / 127` per
/// 32-element block, floored so an all-zero block cannot divide by zero.
///
/// Split out of [`quantize_weight`] so that a caller which produces a row's
/// f32 by some OTHER route - decoding it from a quantized checkpoint block
/// rather than reading it out of a materialized fp32 tensor - can reach the
/// identical packed bytes by construction rather than by reimplementing this
/// arithmetic and hoping it matches. The fold is left-to-right and stays that
/// way: `f32::max` propagates the non-NaN operand, so with a NaN present the
/// ORDER is observable, and two callers that disagree about it would disagree
/// about the scale.
#[inline]
pub fn group_scales(row: &[f32], out: &mut [f32]) {
    debug_assert_eq!(row.len(), out.len() * GROUP, "group_scales: row {} vs {} groups", row.len(), out.len());
    for (g, s) in out.iter_mut().enumerate() {
        let blk = &row[g * GROUP..g * GROUP + GROUP];
        *s = blk.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-8) / 127.0;
    }
}

/// Pack one row to `[k/4]` u32 (4 int8 per word, little-endian along K) given
/// that row's `[k/GROUP]` group scales. The companion of [`group_scales`]; see
/// its note on why these two are shared rather than duplicated.
#[inline]
pub fn pack_row(row: &[f32], scales: &[f32], out: &mut [u32]) {
    debug_assert_eq!(out.len(), scales.len() * WORDS_PER_GROUP, "pack_row: {} words vs {} groups", out.len(), scales.len());
    for (g, word_out) in out.iter_mut().enumerate() {
        // Word `g` covers int8 lanes `4g .. 4g+4`, all inside group `g/8`.
        let inv = 1.0 / scales[g / WORDS_PER_GROUP];
        let mut word = 0u32;
        for b in 0..4 {
            let q = (row[g * 4 + b] * inv).round().clamp(-127.0, 127.0) as i32;
            word |= ((q as u8) as u32) << (8 * b);
        }
        *word_out = word;
    }
}

/// The exact host-side inverse of [`quantize_weight`]: unpack `[n, k/4]` u32
/// words back to `[n, k]` f32 via `sw[r, g]` (the same `[n, k/GROUP]`
/// group-wise scales `quantize_weight` wrote). Used where a real checkpoint
/// quantized a weight (`qwen3omnimoe::import::should_quantize`) that the
/// consuming forward pass still wants as plain f32 - e.g. attention/router
/// projections, which meet the same rank-2/`k % GROUP == 0` shape test as the
/// MoE experts but have no int8 dispatch path of their own (only the experts
/// do, `model::moe::expert_fwd_i8`). `k` must be a multiple of [`GROUP`]
/// (mirrors [`quantize_weight`]'s own requirement).
pub fn dequantize_weight(packed: &[u32], sw: &[f32], n: usize, k: usize) -> Vec<f32> {
    assert_eq!(k % GROUP, 0, "int8 K must be a multiple of {GROUP} (got {k})");
    let kg = k / 4;
    assert_eq!(packed.len(), n * kg, "packed len {} != n*(k/4) {}", packed.len(), n * kg);
    assert_eq!(sw.len(), scale_len(n, k), "scale len {} != n*(k/{GROUP}) {}", sw.len(), scale_len(n, k));
    let mut w = vec![0f32; n * k];
    let gs = k / GROUP;
    for r in 0..n {
        for g in 0..kg {
            let s = sw[r * gs + g / WORDS_PER_GROUP];
            let word = packed[r * kg + g];
            for b in 0..4 {
                let q = ((word >> (8 * b)) as u8) as i8;
                w[r * k + g * 4 + b] = q as f32 * s;
            }
        }
    }
    w
}

/// Dequantize rows `[r0, r0 + rows)` of a packed tensor into `out` (cleared
/// and refilled) - [`dequantize_weight`] restricted to a row block, so a
/// caller can walk a huge tensor without ever holding its f32 expansion.
/// `packed` holds exactly those rows' words (`rows * k/4`); `sw` is the FULL
/// `[n, k/GROUP]` scale tensor, indexed absolutely from row `r0`.
pub fn dequantize_rows_into(packed: &[u32], sw: &[f32], r0: usize, rows: usize, k: usize, out: &mut Vec<f32>) {
    assert_eq!(k % GROUP, 0, "int8 K must be a multiple of {GROUP} (got {k})");
    let kg = k / 4;
    let gs = k / GROUP;
    assert_eq!(packed.len(), rows * kg, "packed len {} != rows*(k/4) {}", packed.len(), rows * kg);
    out.clear();
    out.reserve(rows * k);
    for r in 0..rows {
        let srow = &sw[(r0 + r) * gs..(r0 + r) * gs + gs];
        for g in 0..kg {
            let s = srow[g / WORDS_PER_GROUP];
            let word = packed[r * kg + g];
            for b in 0..4 {
                let q = ((word >> (8 * b)) as u8) as i8;
                out.push(q as f32 * s);
            }
        }
    }
}

/// Stream a packed int8 tensor from `source` onto a device **as f32**, with
/// peak host allocation bounded to one row block rather than the whole
/// dequantized tensor.
///
/// This is the loader for every int8-native weight that has no int8 dispatch
/// path of its own (attention/router projections, `lm_head`): the checkpoint
/// stores them packed because the importer quantizes every rank-2 tensor, but
/// the kernels that consume them are fp32, so they must be unpacked on the way
/// in. Doing that with `tensor_u32` + [`dequantize_weight`] costs the FULL
/// f32 expansion in host RAM first - 1.2 GB for a real `lm_head` at
/// `vocab=152064, hidden=2048`, on top of the packed copy. Here the packed
/// words arrive in bounded chunks (cut on row boundaries so each block's rows
/// are complete), are expanded into one reused scratch, and go straight to
/// the device at their true offset.
///
/// `n`/`k` are the LOGICAL (unpacked) shape; the packed tensor is `[n, k/4]`,
/// so `k` cannot be recovered from its own shape. The group-scale sibling is
/// `"{name}.scale"` (`n * k/GROUP` floats - still small, read whole), checked
/// through [`check_scale_len`] so a pre-group-wise checkpoint fails here with
/// the reason rather than dequantizing to garbage.
pub fn upload_dequantized(
    up: &mut paramstore::upload::Uploader,
    source: &dyn checkpoint::TensorSource,
    name: &str,
    n: usize,
    k: usize,
) -> Result<gpu_core::DeviceBuffer, String> {
    let kg = k / 4;
    if k == 0 || !k.is_multiple_of(GROUP) {
        return Err(format!("upload_dequantized '{name}': k={k} is not a positive multiple of {GROUP}"));
    }
    let scale_name = format!("{name}.scale");
    let mut sw: Vec<f32> = Vec::new();
    if !source.with_tensor(&scale_name, &mut |s| sw = s.to_vec()) {
        return Err(format!("upload_dequantized '{name}': missing scale sibling '{scale_name}'"));
    }
    check_scale_len(&scale_name, sw.len(), n, k).map_err(|e| format!("upload_dequantized '{name}': {e}"))?;

    let buf = up.gpu().storage((n * k) as u64);
    // Cut chunks on ROW boundaries: each callback then holds whole rows, so
    // its scale indices and its destination offset are both exact.
    let rows_per_chunk = (paramstore::UPLOAD_CHUNK_WORDS / kg).max(1);
    let mut scratch: Vec<f32> = Vec::with_capacity(rows_per_chunk * k);
    let mut rows_done = 0usize;
    let mut err: Option<String> = None;
    let found = source.with_tensor_u32_chunks(name, rows_per_chunk * kg, &mut |off_words, chunk| {
        if err.is_some() {
            return;
        }
        if !(off_words as usize).is_multiple_of(kg) || !chunk.len().is_multiple_of(kg) {
            err = Some(format!("upload_dequantized '{name}': chunk at word {off_words} of len {} is not row-aligned (k/4={kg})", chunk.len()));
            return;
        }
        let r0 = off_words as usize / kg;
        let rows = chunk.len() / kg;
        if r0 + rows > n {
            err = Some(format!("upload_dequantized '{name}': rows {r0}..{} exceed the declared {n}", r0 + rows));
            return;
        }
        dequantize_rows_into(chunk, &sw, r0, rows, k, &mut scratch);
        up.gpu().write_f32_at(&buf, (r0 * k) as u64, &scratch);
        rows_done += rows;
    });
    if let Some(e) = err {
        return Err(e);
    }
    if !found {
        return Err(format!("upload_dequantized '{name}': not present as a packed tensor in this source"));
    }
    if rows_done != n {
        return Err(format!("upload_dequantized '{name}': wrote {rows_done} rows, expected {n}"));
    }
    up.account(4 * (n * k) as u64);
    up.maybe_drain(&buf);
    Ok(buf)
}

/// Requantize the rectangle `rows [r0, r0+n_out) x cols [c0, c0+k)` of a
/// tensor stored as `[_, stride]` straight from Q8_0 blocks to
/// `(packed, scales)` - a byte repack, not a dequantize-then-requantize, now
/// that [`GROUP`] (32) matches Q8_0's own block size exactly: a Q8_0 block
/// stores `d = max|x|/127`, `q = round(x/d)`, so `max|q| = 127`, the group's
/// own absmax is `127*d`, and `group_scales` reproduces `d` bit-exactly -
/// every `q` requantizes to itself. `deq_q8_0`-shaped decode needs at most 18
/// significand bits (7 for `q`, 11 for the fp16 scale) against fp32's 24, so
/// the f32 the round trip would have quantized is reproduced exactly.
///
/// `None` means "use the fp32 route": `source` has no quantized blocks for
/// `name` (declines from [`checkpoint::TensorSource::raw_blocks`] - not a
/// GGUF source, not Q8_0, or a wrapper with a value transform in the way,
/// see `qwen35::int8_gguf_resident::SsmALogFix`), or the rectangle's bounds
/// are not Q8_0-block-aligned (`stride`/`c0`/`k` each a multiple of 32).
/// Returning `None` is always safe - the caller's fp32 fallback produces the
/// same bytes by the longer route. A caller that must exclude a tensor for a
/// reason of its own (a LoRA fold needing a float domain) checks that BEFORE
/// calling this, by having its own `TensorSource` decline first - not this
/// function's concern.
///
/// The one canonical implementation: `gguf::int8_direct::try_i8_rect` is a
/// thin wrapper over this (that crate depends on `model`, not the reverse,
/// so the algorithm has to live here to be shared).
pub fn q8_0_rect(
    source: &dyn checkpoint::TensorSource,
    name: &str,
    stride: usize,
    r0: usize,
    n_out: usize,
    c0: usize,
    k: usize,
) -> Option<(Vec<u32>, Vec<f32>)> {
    use checkpoint::gguf::{block_expand, GgmlType, Q8_0_BLOCK_ELEMS};
    if !stride.is_multiple_of(Q8_0_BLOCK_ELEMS) || !c0.is_multiple_of(Q8_0_BLOCK_ELEMS) || !k.is_multiple_of(Q8_0_BLOCK_ELEMS) {
        return None;
    }
    let (layout, raw) = source.raw_blocks(name)?;
    if layout.ty != GgmlType::Q8_0 {
        return None;
    }
    let kg = k / 4;
    let gs = k / GROUP;
    let mut packed = vec![0u32; n_out * kg];
    let mut sw = vec![0f32; n_out * gs];
    // One output row per task: each reads only its own block range and
    // writes only its own words and scales, so this is bit-identical to the
    // serial form and to the fp32 round trip alike.
    backend_cpu::par::chunks2_mut(&mut packed, kg, &mut sw, gs, |i, prow, srow| {
        let mut row = Vec::with_capacity(k);
        let e0 = (r0 + i) * stride + c0;
        block_expand(GgmlType::Q8_0, raw, e0, e0 + k, &mut row).expect("block-aligned above");
        group_scales(&row, srow);
        pack_row(&row, srow, prow);
    });
    Some((packed, sw))
}

/// Host-only quantization of a WHOLE `[n, k]` linear from `source`, choosing
/// the cheapest route it can serve: [`q8_0_rect`]'s byte repack (no fp32
/// anywhere) when the source offers Q8_0 blocks, else a bounded fp32 route -
/// `with_tensor_chunks` cut on ROW boundaries so peak host allocation is one
/// row block, never `n*k` - through [`quantize_weight`]. `None` if `name` is
/// not present in `source` at all.
///
/// For a caller that stops before the device (a host cache the model
/// re-uploads from later, e.g. a checkpoint-quantized weight kept resident
/// as bytes rather than re-read every build).
pub fn quantize_from(source: &dyn checkpoint::TensorSource, name: &str, n: usize, k: usize) -> Option<(Vec<u32>, Vec<f32>)> {
    if let Some(direct) = q8_0_rect(source, name, k, 0, n, 0, k) {
        return Some(direct);
    }
    assert!(k.is_multiple_of(GROUP), "quantize_from '{name}': k must be a multiple of {GROUP} (got {k})");
    let kg = k / 4;
    let gs = k / GROUP;
    let mut packed = vec![0u32; n * kg];
    let mut sw = vec![0f32; n * gs];
    let mut any = false;
    // Cut on ROW boundaries (a multiple of k, never a bare k) so every
    // callback holds whole rows - its destination offset (`row0 * kg` /
    // `row0 * gs`) is exact without tracking a running remainder - while
    // still batching `UPLOAD_CHUNK_WORDS`-ish elements per call rather than
    // one row at a time. `qwen35::stream::quantize_i8_rows` (the code this
    // replaces) let each caller tune its own rows-per-chunk; this picks one
    // constant for every caller instead, matching `paramstore::upload::
    // Uploader`'s own chunk sizing rather than adding a second tunable.
    let rows_per_chunk = (paramstore::UPLOAD_CHUNK_WORDS / k).max(1);
    let found = source.with_tensor_chunks(name, rows_per_chunk * k, &mut |off, chunk| {
        any = true;
        assert_eq!(off as usize % k, 0, "quantize_from '{name}': chunk offset {off} is not row-aligned (k={k})");
        assert_eq!(chunk.len() % k, 0, "quantize_from '{name}': chunk length {} is not a whole number of rows (k={k})", chunk.len());
        let rows = chunk.len() / k;
        let row0 = off as usize / k;
        let (rpacked, rsw) = quantize_weight(chunk, rows, k);
        packed[row0 * kg..(row0 + rows) * kg].copy_from_slice(&rpacked);
        sw[row0 * gs..(row0 + rows) * gs].copy_from_slice(&rsw);
    });
    if !found || !any {
        return None;
    }
    Some((packed, sw))
}

/// [`quantize_from`]'s generalization to a rectangle - the shape
/// [`upload_rect`] needs for a fused `qkv`/`linear1` split. Delegates to
/// [`quantize_from`] for the "whole tensor" rectangle (`stride == k`,
/// `r0 == 0`, `c0 == 0`) so that common case keeps the bounded fp32
/// fallback; a genuine sub-rectangle (a row slice, a column block of a fused
/// matrix) still tries [`q8_0_rect`] first, then falls back to materializing
/// the whole tensor as fp32 and slicing the rectangle out of it - the same
/// cost `flux2::weights::DitWeights::with_f32` already pays for a rect
/// fallback, not a new one this introduces.
pub fn quantize_rect_from(
    source: &dyn checkpoint::TensorSource,
    name: &str,
    stride: usize,
    r0: usize,
    n_out: usize,
    c0: usize,
    k: usize,
) -> Option<(Vec<u32>, Vec<f32>)> {
    if stride == k && r0 == 0 && c0 == 0 {
        return quantize_from(source, name, n_out, k);
    }
    if let Some(direct) = q8_0_rect(source, name, stride, r0, n_out, c0, k) {
        return Some(direct);
    }
    let mut whole: Vec<f32> = Vec::new();
    if !source.with_tensor(name, &mut |d| whole = d.to_vec()) {
        return None;
    }
    let mut rect = Vec::with_capacity(n_out * k);
    for r in 0..n_out {
        let row_start = (r0 + r) * stride + c0;
        rect.extend_from_slice(&whole[row_start..row_start + k]);
    }
    Some(quantize_weight(&rect, n_out, k))
}

/// Upload `name`'s `[n, k]` linear onto the device as packed int8 + group
/// scales, via [`quantize_from`]'s cheapest-route choice.
pub fn upload_quantized(
    up: &mut paramstore::upload::Uploader,
    source: &dyn checkpoint::TensorSource,
    name: &str,
    n: usize,
    k: usize,
) -> Result<(gpu_core::DeviceBuffer, gpu_core::DeviceBuffer), String> {
    let (packed, sw) = quantize_from(source, name, n, k).ok_or_else(|| format!("upload_quantized '{name}': not present in this source"))?;
    upload_packed_scales(up, packed, sw)
}

/// [`upload_quantized`] over a rectangle - see [`quantize_rect_from`].
#[allow(clippy::too_many_arguments)]
pub fn upload_rect(
    up: &mut paramstore::upload::Uploader,
    source: &dyn checkpoint::TensorSource,
    name: &str,
    stride: usize,
    r0: usize,
    n_out: usize,
    c0: usize,
    k: usize,
) -> Result<(gpu_core::DeviceBuffer, gpu_core::DeviceBuffer), String> {
    let (packed, sw) =
        quantize_rect_from(source, name, stride, r0, n_out, c0, k).ok_or_else(|| format!("upload_rect '{name}': not present in this source"))?;
    upload_packed_scales(up, packed, sw)
}

/// Shared tail of [`upload_quantized`]/[`upload_rect`]: land already-computed
/// packed words + group scales on the device, under the uploader's staging
/// discipline (`account`/`maybe_drain` - see `paramstore::upload::Uploader`'s
/// own doc on why a caller writing through `Gpu::write*_at` directly still
/// needs this).
fn upload_packed_scales(
    up: &mut paramstore::upload::Uploader,
    packed: Vec<u32>,
    sw: Vec<f32>,
) -> Result<(gpu_core::DeviceBuffer, gpu_core::DeviceBuffer), String> {
    let wbuf = up.gpu().storage(packed.len() as u64);
    up.gpu().write_at(&wbuf, 0, &packed);
    let sbuf = up.gpu().storage(sw.len() as u64);
    up.gpu().write_f32_at(&sbuf, 0, &sw);
    up.account(4 * (packed.len() + sw.len()) as u64);
    up.maybe_drain(&wbuf);
    Ok((wbuf, sbuf))
}

/// The buffers and kernel slots one dynamic activation quantization needs —
/// the same "bundle the ids so the call stays readable" shape as
/// [`crate::block::FlashIds`].
pub struct QuantRows<'a> {
    /// `[max_abs_row, quant_pack]` in the caller's kernel list.
    pub kernels: [usize; 2],
    /// `[.., k]` f32 activation to quantize.
    pub x: &'a gpu_core::DeviceBuffer,
    /// `[rows]` per-token scale (written).
    pub sx: &'a gpu_core::DeviceBuffer,
    /// `[.., k/4]` packed u32 activation (written).
    pub xq: &'a gpu_core::DeviceBuffer,
    /// Third, OPTIONAL step: `(quant_group_sum kernel index, [.., k/32] f32
    /// output buffer)`. `Some` only for a caller building an affine K-quant
    /// activation (`crate::kquant`'s `S[m,g] = Σ_{k in g} xq[m,k]` correction
    /// term) - `None` for every existing caller, whose dispatch stays the
    /// existing two-step `[max_abs_row, quant_pack]` form byte-identically.
    pub xgs: Option<(usize, &'a gpu_core::DeviceBuffer)>,
}

/// The DEVICE half of the same tier: dynamic per-token activation quantization
/// of rows `r0..r1` of `q.x` — `max_abs_row` writes `q.sx[r0..r1]`,
/// `quant_pack` writes the packed rows of `q.xq`. Returns the steps in order;
/// ONE call feeds every linear that reads that activation.
///
/// This exists so the *offset units* live in one place. Every buffer is bound
/// as a sub-range and the units differ per buffer — `x` and `xq` are offset in
/// ELEMENTS of their own width (`k` vs `k/4`) while `sx` is offset in ROWS.
/// Getting one wrong is silently wrong arithmetic, not a crash; `step_sliced`'s
/// element-vs-byte contract already cost this repo a SIGSEGV.
///
/// **Alignment:** `step_sliced` turns each offset into a real
/// `BufferBinding::offset`, so every one must clear
/// `min_storage_buffer_offset_alignment` (256 B = 64 floats on a P40). `k` is
/// normally a multiple of 64, so `x`/`xq` are safe for any `r0`; `sx` is offset
/// by `r0` itself, so **`r0` must be a multiple of 64** — asserted here rather
/// than left to each caller to remember (a violation is a wgpu validation
/// error, not a wrong number, so it hides until someone changes a text length).
///
/// When `q.xgs` is `Some((kernel, xgs))`, a THIRD step (`quant_group_sum`,
/// `crate::kquant`'s `S[m,g] = Σ_{k in g} xq[m,k]` correction term) is
/// appended, reading the SAME `xq` rows the second step just wrote. `xgs` is
/// offset in ROWS of `k/32` (its own width, one f32 per 32-element group) -
/// the same "own width" convention `x`/`xq` already follow, just at group
/// granularity instead of element or row-count granularity. `q.xgs` being
/// `None` is byte-identical to this function's original two-step form - no
/// existing caller's dispatch changes.
pub fn quant_rows_steps(g: &gpu_core::Gpu, q: QuantRows, r0: u32, r1: u32, k: u32) -> Vec<gpu_core::Step> {
    assert_eq!(k % 4, 0, "int8 K must be a multiple of 4 (got {k})");
    assert!(
        r0.is_multiple_of(64),
        "int8 activation quant: row base {r0} breaks the 64-float storage-binding alignment of the per-token scale buffer"
    );
    let m = r1 - r0;
    let xo = (r0 as u64 * k as u64, m as u64 * k as u64);
    let so = (r0 as u64, m as u64);
    let qo = (r0 as u64 * (k as u64 / 4), m as u64 * (k as u64 / 4));
    let mut steps = vec![
        g.step_sliced(q.kernels[0], &[q.x, q.sx], &[xo, so], &[m, k], m),
        g.step_sliced(q.kernels[1], &[q.x, q.sx, q.xq], &[xo, so, qo], &[m, k], m * k / 4),
    ];
    if let Some((kernel, xgs)) = q.xgs {
        assert_eq!(k % GROUP as u32, 0, "int8 K-quant group sum: K must be a multiple of {GROUP} (got {k})");
        let go = (r0 as u64 * (k as u64 / GROUP as u64), m as u64 * (k as u64 / GROUP as u64));
        steps.push(g.step_sliced(kernel, &[q.xq, xgs], &[qo, go], &[m, k], m * k / GROUP as u32));
    }
    steps
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scale that governs element `c` of row `r`.
    fn scale_at(sw: &[f32], k: usize, r: usize, c: usize) -> f32 {
        sw[r * (k / GROUP) + c / GROUP]
    }

    #[test]
    fn round_trips_within_one_step() {
        let (n, k) = (3, 64);
        let w: Vec<f32> = (0..n * k).map(|i| (i as f32 - 10.0) * 0.37).collect();
        let (packed, sw) = quantize_weight(&w, n, k);
        assert_eq!(packed.len(), n * k / 4);
        assert_eq!(sw.len(), scale_len(n, k));
        for r in 0..n {
            for c in 0..k {
                let word = packed[r * (k / 4) + c / 4];
                let q = ((word >> (8 * (c % 4))) & 0xff) as u8 as i8;
                let s = scale_at(&sw, k, r, c);
                let deq = q as f32 * s;
                assert!((deq - w[r * k + c]).abs() <= s * 0.5 + 1e-6, "r{r} c{c}");
            }
        }
    }

    /// [`quantize_weight`] must reject a `k` that is not a whole number of
    /// [`GROUP`]s rather than quantize a ragged tail - `Q8_0`'s own rule, and
    /// the reason every caller's shape test is `k % 32 == 0`.
    #[test]
    #[should_panic(expected = "multiple of 32")]
    fn quantize_weight_rejects_a_ragged_k() {
        quantize_weight(&[0.5f32; 2 * 12], 2, 12);
    }

    /// The whole point of the group-wise layout, stated as a measurement.
    ///
    /// One row of otherwise-small weights with a single large outlier is the
    /// exact failure mode `AGENTS.md` records for whole-channel scales: the
    /// outlier sets `max|row|`, so every small weight in that ROW quantizes
    /// against a step ~300x too coarse and most of them collapse to code 0.
    /// Group-wise, the damage is CONFINED to the outlier's own 32-element
    /// block; the other `k/32 - 1` blocks keep their own resolution.
    ///
    /// Two things are asserted, because either alone is weak: the error on the
    /// elements OUTSIDE the outlier's block (where group-wise must be orders
    /// of magnitude better), and the COUNT of badly-quantized elements in the
    /// whole tensor (where group-wise must be bounded by the block size,
    /// whatever `k` is - that boundedness is the property, not a number).
    ///
    /// The whole-channel figures are computed here from the same weights via
    /// the old formula (`max|row|/127`, one scale for the row), so the
    /// comparison is a measurement against the scheme actually replaced,
    /// not against a remembered constant.
    #[test]
    fn group_wise_confines_a_row_outlier_that_whole_channel_spreads_over_the_row() {
        let (n, k) = (2, 512);
        // Small, structured, non-degenerate values plus one ~300x outlier per
        // row, placed in the LAST group so most of the row is far from it.
        let mut w: Vec<f32> = (0..n * k).map(|i| (i % 37) as f32 * 0.017 - 0.3).collect();
        let outlier = k - 3;
        for r in 0..n {
            w[r * k + outlier] = 100.0;
        }
        let (packed, sw) = quantize_weight(&w, n, k);
        let deq = dequantize_weight(&packed, &sw, n, k);

        // The whole-channel scheme this replaced: one scale per row.
        let whole: Vec<f32> = (0..n).map(|r| w[r * k..r * k + k].iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-8) / 127.0).collect();
        let whole_deq = |r: usize, c: usize| (w[r * k + c] / whole[r]).round().clamp(-127.0, 127.0) * whole[r];

        // (a) worst error OUTSIDE the outlier's own 32-element block.
        let (mut group_max, mut whole_max) = (0f32, 0f32);
        // (b) how many elements are quantized to worse than a third of the
        //     typical |w| here (~0.3) - i.e. effectively destroyed.
        let (mut group_bad, mut whole_bad) = (0usize, 0usize);
        for r in 0..n {
            for c in 0..k {
                let orig = w[r * k + c];
                let ge = (deq[r * k + c] - orig).abs();
                let we = (whole_deq(r, c) - orig).abs();
                if c / GROUP != outlier / GROUP {
                    group_max = group_max.max(ge);
                    whole_max = whole_max.max(we);
                }
                if c != outlier {
                    group_bad += usize::from(ge > 0.1);
                    whole_bad += usize::from(we > 0.1);
                }
            }
        }
        eprintln!(
            "row outlier at k-3: outside its block, max_abs group-wise {group_max:.6} vs whole-channel {whole_max:.6} ({:.0}x); \
             elements destroyed (err > 0.1): {group_bad} vs {whole_bad} of {}",
            whole_max / group_max,
            n * (k - 1)
        );
        assert!(group_max < 0.005, "group-wise max_abs {group_max} outside the outlier's block");
        assert!(whole_max > 0.2, "whole-channel max_abs {whole_max} - the fixture must actually crush the row");
        assert!(whole_max / group_max > 50.0, "group-wise {group_max} vs whole-channel {whole_max}");
        // Bounded by the outlier's own block, per row - not by `k`.
        assert!(group_bad <= n * (GROUP - 1), "group-wise damage must stay inside the outlier's block: {group_bad} elements");
        assert!(whole_bad > n * (k - 1) / 2, "whole-channel must destroy most of the row, got {whole_bad}");
    }

    /// [`dequantize_weight`] must agree with the round-trip math
    /// [`round_trips_within_one_step`] checks inline, per-element, over
    /// multiple rows and several groups per row (so a scale-indexing slip
    /// between `[n]` and `[n, k/32]` shows up) - the real caller of this
    /// function (`qwen3omnimoe::int8_thinker_resident::load_mat`) needs it for
    /// real weight shapes, not just the one `quantize_weight`'s own test
    /// happens to use.
    #[test]
    fn dequantize_weight_matches_quantize_weight_within_one_step() {
        let (n, k) = (5, 96);
        let w: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.11 - 6.0) * (1 + i % 3) as f32).collect();
        let (packed, sw) = quantize_weight(&w, n, k);
        let deq = dequantize_weight(&packed, &sw, n, k);
        assert_eq!(deq.len(), n * k);
        for r in 0..n {
            for c in 0..k {
                let d = deq[r * k + c];
                let orig = w[r * k + c];
                let s = scale_at(&sw, k, r, c);
                assert!((d - orig).abs() <= s * 0.5 + 1e-6, "r{r} c{c}: deq={d} orig={orig} scale={s}");
            }
        }
    }

    /// A whole-channel (`[n]`) scale tensor must be REJECTED by name, not read
    /// as if it were the first row's worth of group scales.
    #[test]
    fn a_whole_channel_scale_tensor_is_rejected_with_the_format_change_named() {
        let (n, k) = (4, 128);
        let e = check_scale_len("blk.0.ffn_down.weight", n, n, k).unwrap_err();
        assert!(e.contains("WHOLE-CHANNEL"), "{e}");
        assert!(e.contains("re-imported"), "{e}");
        check_scale_len("blk.0.ffn_down.weight", scale_len(n, k), n, k).unwrap();
    }

    /// Mutation-verify: an intentionally wrong sign-extension (`as u8`
    /// dropped, reading the byte as unsigned) must break the round-trip
    /// tolerance above -- confirms the test actually exercises the sign
    /// handling, not just magnitude.
    #[test]
    fn dequantize_weight_sign_handling_is_load_bearing() {
        let (n, k) = (2, 32);
        let w: Vec<f32> = (0..n * k).map(|i| if i % 2 == 0 { -(i as f32) - 1.0 } else { i as f32 + 1.0 }).collect();
        let (packed, sw) = quantize_weight(&w, n, k);
        let correct = dequantize_weight(&packed, &sw, n, k);
        // Deliberately wrong: treat each packed byte as UNSIGNED, dropping
        // the `as i8` reinterpretation `dequantize_weight` relies on.
        let mut wrong = vec![0f32; n * k];
        for r in 0..n {
            for g in 0..k / 4 {
                let s = sw[r * (k / GROUP) + g / WORDS_PER_GROUP];
                let word = packed[r * (k / 4) + g];
                for b in 0..4 {
                    let q_unsigned = (word >> (8 * b)) as u8;
                    wrong[r * k + g * 4 + b] = q_unsigned as f32 * s;
                }
            }
        }
        let mut any_diverged = false;
        for i in 0..w.len() {
            if (correct[i] - wrong[i]).abs() > 1e-3 {
                any_diverged = true;
            }
        }
        assert!(any_diverged, "unsigned-vs-signed byte reinterpretation should diverge for at least one negative-quantized element");
    }

    /// A GGUF-backed source with one Q8_0 tensor, built through
    /// `checkpoint::quant`'s real encoder - not hand-assembled bytes.
    fn q8_0_source(name: &str, vals: &[f32]) -> (checkpoint::gguf::MmapGguf, String) {
        // A process-wide counter, not just the pid: several tests in this
        // file build a fixture named "w" and run concurrently under cargo's
        // default multi-threaded test runner, so pid+name alone collided
        // (one test's write raced another's cleanup unlink on the identical
        // path).
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let block = checkpoint::quant::quantize_par(checkpoint::gguf::TYPE_Q8_0, vals).unwrap();
        let path = std::env::temp_dir().join(format!("brain-model-int8-{}-{}-{id}.gguf", name.replace('.', "_"), std::process::id()));
        let path = path.to_str().unwrap().to_string();
        checkpoint::gguf_write::write(
            &path,
            &[],
            &[checkpoint::gguf_write::TensorOut { name: name.to_string(), shape: vec![vals.len()], ty: checkpoint::gguf::TYPE_Q8_0, data: block }],
            32,
        )
        .unwrap();
        (checkpoint::gguf::MmapGguf::open(&path).unwrap(), path)
    }

    /// [`q8_0_rect`] must be bit-identical to the fp32 round trip
    /// (`quantize_weight` over the dequantized values), the same claim
    /// `flux2/tests/gguf_direct_int8.rs` proves for the crate this shares its
    /// algorithm with - checked here so `model::int8` owning the
    /// implementation doesn't lose the property.
    #[test]
    fn q8_0_rect_is_bit_identical_to_the_fp32_round_trip() {
        let (n, k) = (3usize, 64usize);
        let vals: Vec<f32> = (0..n * k).map(|i| ((i as i64 * 37 - 511) % 251) as f32 * 0.5).collect();
        let (mg, path) = q8_0_source("w", &vals);

        let direct = q8_0_rect(&mg, "w", k, 0, n, 0, k).expect("Q8_0 source, block-aligned rect must take the direct path");
        let deq = mg.tensor("w").unwrap().unwrap();
        let want = quantize_weight(&deq, n, k);
        assert_eq!(direct, want, "byte repack must match the fp32 round trip bit-for-bit");

        // Row offset r0 != 0: the case a naive implementation gets wrong.
        let row1 = q8_0_rect(&mg, "w", k, 1, 1, 0, k).unwrap();
        let want_row1 = quantize_weight(&deq[k..2 * k], 1, k);
        assert_eq!(row1, want_row1, "a nonzero row offset must land on the right block range");

        std::fs::remove_file(&path).ok();
    }

    /// Alignment refusals decline rather than silently widen or misalign.
    #[test]
    fn q8_0_rect_declines_unaligned_bounds() {
        let (n, k) = (2usize, 64usize);
        let vals: Vec<f32> = (0..n * k).map(|i| i as f32).collect();
        let (mg, path) = q8_0_source("w", &vals);
        assert!(q8_0_rect(&mg, "w", 64, 0, 2, 0, 20).is_none(), "k=20 is not block-aligned");
        assert!(q8_0_rect(&mg, "w", 64, 0, 2, 20, 32).is_none(), "c0=20 is not block-aligned");
        assert!(q8_0_rect(&mg, "w", 50, 0, 2, 0, 50).is_none(), "stride=50 is not block-aligned");
        // A non-Q8_0-typed name (a plain HashMap source) declines too.
        let mut m: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
        m.insert("w".to_string(), vals);
        assert!(q8_0_rect(&m, "w", 64, 0, 2, 0, 64).is_none(), "a source with no raw_blocks must decline, not panic");
        std::fs::remove_file(&path).ok();
    }

    /// [`quantize_from`] takes the direct route when the source offers Q8_0
    /// blocks and the bounded fp32 route otherwise, and both land on the
    /// SAME values for the same logical weight - proven by comparing a
    /// Q8_0-backed source against a plain f32 `HashMap` holding the
    /// tensor's OWN dequantized values (the honest oracle: Q8_0 quantization
    /// is lossy, so the two inputs are not bit-identical, but re-quantizing
    /// the dequantized f32 through the SAME group-wise math must reproduce
    /// exactly what the direct byte repack already proved above).
    #[test]
    fn quantize_from_matches_across_both_routes() {
        let (n, k) = (4usize, 96usize);
        let vals: Vec<f32> = (0..n * k).map(|i| ((i as i64 * 13 - 777) % 401) as f32 * 0.25).collect();
        let (mg, path) = q8_0_source("blk.0.weight", &vals);
        let via_gguf = quantize_from(&mg, "blk.0.weight", n, k).expect("present via the direct route");

        let deq = mg.tensor("blk.0.weight").unwrap().unwrap();
        let mut m: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
        m.insert("blk.0.weight".to_string(), deq.clone());
        let via_fallback = quantize_from(&m, "blk.0.weight", n, k).expect("present via the fp32 fallback route");

        assert_eq!(via_gguf, via_fallback, "both routes must land on the identical packed weight for the same logical tensor");
        assert_eq!(via_gguf, quantize_weight(&deq, n, k), "and both must agree with quantize_weight itself");

        assert!(quantize_from(&mg, "does.not.exist", n, k).is_none());
        std::fs::remove_file(&path).ok();
    }

    /// [`quantize_rect_from`] on a genuine sub-rectangle (mid-tensor row
    /// range AND a column offset, so neither bound is 0) must match slicing
    /// the rectangle out of the fully dequantized tensor and quantizing that
    /// - exercised over an f32 `HashMap` source (no Q8_0 blocks to align
    /// against), which forces the whole-tensor-materialize-and-slice
    /// fallback rather than the direct route.
    #[test]
    fn quantize_rect_from_a_genuine_subrectangle_matches_manual_slicing() {
        let (rows, cols) = (6usize, 96usize);
        let whole: Vec<f32> = (0..rows * cols).map(|i| (i as f32).sin() * 10.0).collect();
        let mut m: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
        m.insert("fused".to_string(), whole.clone());

        // k and cols must each be multiples of GROUP (32) - quantize_weight's
        // own requirement, which the fp32 fallback this test exercises
        // inherits unchanged.
        let (r0, n_out, c0, k) = (2usize, 3usize, 32usize, 32usize);
        let got = quantize_rect_from(&m, "fused", cols, r0, n_out, c0, k).expect("present");

        let mut manual = Vec::with_capacity(n_out * k);
        for r in 0..n_out {
            let start = (r0 + r) * cols + c0;
            manual.extend_from_slice(&whole[start..start + k]);
        }
        let want = quantize_weight(&manual, n_out, k);
        assert_eq!(got, want);

        // The whole-tensor rectangle (stride==k, r0=c0=0) must delegate to
        // quantize_from and therefore match it exactly.
        let mut whole_m: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
        whole_m.insert("w".to_string(), whole[..cols * 2].to_vec());
        let rect_whole = quantize_rect_from(&whole_m, "w", cols, 0, 2, 0, cols).unwrap();
        let from_whole = quantize_from(&whole_m, "w", 2, cols).unwrap();
        assert_eq!(rect_whole, from_whole);
    }

    /// [`upload_quantized`]/[`upload_rect`] must actually land the SAME
    /// bytes [`quantize_from`]/[`quantize_rect_from`] compute, read back
    /// from the device - not just call them and trust the upload.
    #[test]
    fn upload_quantized_and_upload_rect_read_back_correctly() {
        let g = gpu_core::testgpu::dev(&[]);
        let mut up = paramstore::upload::Uploader::new(&g);

        let (n, k) = (2usize, 64usize);
        let vals: Vec<f32> = (0..n * k).map(|i| ((i as i64 * 29 - 300) % 199) as f32 * 0.3).collect();
        let (mg, path) = q8_0_source("w", &vals);

        // `Gpu::read` only ever reads back f32; a packed-u32 buffer's words
        // were written bit-for-bit via `write_at` -> `write_f32_at`'s cast
        // sibling, so reading it back as f32 and taking `to_bits()` recovers
        // the original u32 words exactly (the same trick `write_f32_at`
        // itself uses in reverse).
        let read_u32 = |buf: &gpu_core::DeviceBuffer, n: usize| -> Vec<u32> { g.read(buf, n).into_iter().map(f32::to_bits).collect() };

        let (want_packed, want_sw) = quantize_from(&mg, "w", n, k).unwrap();
        let (wbuf, sbuf) = upload_quantized(&mut up, &mg, "w", n, k).unwrap();
        assert_eq!(read_u32(&wbuf, want_packed.len()), want_packed, "upload_quantized must write exactly what quantize_from computed");
        assert_eq!(g.read(&sbuf, want_sw.len()), want_sw);

        let (r0, n_out, c0, kk) = (1usize, 1usize, 0usize, k);
        let (want_rpacked, want_rsw) = quantize_rect_from(&mg, "w", k, r0, n_out, c0, kk).unwrap();
        let (rwbuf, rsbuf) = upload_rect(&mut up, &mg, "w", k, r0, n_out, c0, kk).unwrap();
        assert_eq!(read_u32(&rwbuf, want_rpacked.len()), want_rpacked);
        assert_eq!(g.read(&rsbuf, want_rsw.len()), want_rsw);

        std::fs::remove_file(&path).ok();
    }
}

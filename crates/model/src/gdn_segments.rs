// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Gated DeltaNet mixer over all the prompt segments of one serving round in a single pass per layer.
//!
//! Swedish Embedded AB implements solutions for serving many concurrent requests on one accelerator for its clients. If
//! your team needs expertise in batching the recurrent layers of a language model across requests without a launch per
//! request, then you can procure our services by sending an email to info@swedishembedded.com.
//!
//! A mixed serving round carries several requests' prompt rows (and decode rows) through one forward. The per-segment
//! path ([`crate::gdn_mixer::gdn_mixer_stream_fwd`]) runs every segment, and every full-chunk / short-remainder piece of
//! it, as its own convolution, prep and recurrence launch on copies of its rows, with the state gathered from the pool
//! before and scattered back after: about a dozen latency-bound launches a piece. Here the whole round is one launch of
//! each of three kernels per layer (convolution, chunk prep, recurrence), addressed by a table built once per round:
//!
//! * the segments' rows are read and written in place in the round's token-major tensors (no row copies);
//! * the recurrent state and the convolution history are read from, and left in, each segment's row of the pools (no
//!   gather or scatter);
//! * a segment's last, short chunk is one more chunk of the same walk, not a launch of its own;
//! * the elementwise gates, the gated norm and the SiLU of `z` run once over all the round's rows.
//!
//! Rows are bit-identical to the per-segment path (the same chunk arithmetic in the same order). Served only where the
//! bf16 prefill form is the device's selected candidate; [`GdnSegmentRound::new`] and [`gdn_mixer_segments_fwd`] return
//! `None` otherwise and the caller keeps the per-segment path.

use gpu_core::{DeviceBuffer, Gpu};

use crate::gdn::{tokens_prefix_offered, GdnShape};
use crate::gdn_mixer::{GdnMixerIds, GdnMixerShape, GdnMixerWeights};

/// One prompt segment of a round: `rows` rows from `row0` of the round's activations, continuing pool row `slot`.
#[derive(Clone, Copy, Debug)]
pub struct SegmentSpan {
    pub row0: u32,
    pub rows: u32,
    pub slot: u32,
}

/// A round's segment table, on the device, shared by all its layers.
pub struct GdnSegmentRound {
    #[cfg(not(target_arch = "wasm32"))]
    table: gpu_core::provider::cuda::GdnSegmentTable,
    #[cfg(not(target_arch = "wasm32"))]
    buf: DeviceBuffer,
}

/// The state and history pools of one layer, in rows of `[heads, dk, dv]` and `[conv_dim, kernel - 1]` f32 words.
pub struct GdnSegmentPools<'a> {
    pub state: &'a DeviceBuffer,
    pub hist: &'a DeviceBuffer,
    /// The pools hold bf16 elements (two to a word) rather than f32 words: the kernels widen on load and round to nearest even on store.
    pub bf16: bool,
}

#[cfg(not(target_arch = "wasm32"))]
impl GdnSegmentRound {
    /// The table for `spans`, or `None` when this device does not select the bf16 prefill form (the only form the segment
    /// kernels implement) or has no tensor-core kernels.
    pub fn new(g: &Gpu, spans: &[SegmentSpan]) -> Option<Self> {
        use gpu_core::provider::cuda::{gdn_segment_kernels_served, GdnSegmentSpan, GdnSegmentTable};
        if spans.is_empty() || !gdn_segment_kernels_served(g) {
            return None;
        }
        let spans: Vec<_> = spans.iter().map(|s| GdnSegmentSpan { row0: s.row0, rows: s.rows, slot: s.slot }).collect();
        let table = GdnSegmentTable::new(&spans);
        let buf = g.storage_uninit(table.words().len() as u64);
        g.write(&buf, table.words());
        Some(Self { table, buf })
    }
}

#[cfg(target_arch = "wasm32")]
impl GdnSegmentRound {
    pub fn new(_g: &Gpu, _spans: &[SegmentSpan]) -> Option<Self> {
        None
    }
}

/// What [`gdn_mixer_segments_fwd`] hands the output projection.
pub struct SegmentsOut {
    /// `[n, value_dim]`, the rows `lead ..` written.
    pub gated: DeviceBuffer,
    /// With `e4m3` asked: the E4M3 bytes (`n * value_dim / 4` words) and the scale per row per 128 of `gated`.
    pub e4m3: Option<(DeviceBuffer, DeviceBuffer)>,
}

/// The mixer of one layer over every segment of `round`, from the layer's projections `mixed_qkv`, `bproj`, `aproj`, `z`
/// (all `[n, ...]` for the round's `n` rows) to `gated` (`[n, value_dim]`). Rows not in a segment (a mixed round's
/// leading `lead` decode rows) are left untouched for the caller to fill. `None` (nothing submitted, the pools untouched)
/// when the device or the shape is not served. With `e4m3`, the E4M3 quantisation of `gated` for FP8 products comes with it.
///
/// With `bf16` the round's activations move as bf16 rows: `mixed_qkv` and `z` come in as bf16 (the FP8 projections' bf16
/// output), the convolution's query, key and value and the recurrence output are bf16, and since the E4M3 rows are what the
/// output projection reads (`e4m3` is required) the fp32 `gated` rows of the segments are not stored - only the leading decode
/// rows' are, by the caller. Every kernel still accumulates in f32.
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::too_many_arguments)]
pub fn gdn_mixer_segments_fwd(
    g: &Gpu,
    ids: &GdnMixerIds,
    shape: &GdnMixerShape,
    w: &GdnMixerWeights,
    round: &GdnSegmentRound,
    pools: &GdnSegmentPools,
    mixed_qkv: &DeviceBuffer,
    bproj: &DeviceBuffer,
    aproj: &DeviceBuffer,
    z: &DeviceBuffer,
    n: u32,
    lead: u32,
    e4m3: bool,
    bf16: bool,
) -> Option<SegmentsOut> {
    use gpu_core::provider::cuda::{gdn_chunk_loop_seg_bf16_step, gdn_chunk_prep_seg_bf16_step, gdn_conv_silu_seg_step, gdn_gated_norm_step, GdnConvLayout, GdnTokenLayout};

    let gdn = shape.gdn;
    let (nvh, khd, vhd) = (gdn.h, gdn.dk, gdn.dv);
    let (conv_dim, key_dim, value_dim, group) = (shape.conv_dim(), shape.key_dim(), shape.value_dim(), shape.group());
    let chunk_shape = GdnShape { b: 1, h: nvh, t: n, dk: khd, dv: vhd, chunk: 64 };
    if !tokens_prefix_offered(g, &ids.chunk, &chunk_shape, group) || shape.conv_kernel != 4 {
        return None;
    }
    let table = &round.table;
    if bf16 && !e4m3 {
        return None;
    }
    let slots = (table.chunks() * nvh) as u64 * 64;
    // The convolution's outputs: two bytes an element (words of two) with bf16.
    let words = |elems: u64| if bf16 { elems.div_ceil(2) } else { elems };
    let (query, key, value) = (g.storage_uninit(words(n as u64 * key_dim as u64)), g.storage_uninit(words(n as u64 * key_dim as u64)), g.storage_uninit(words(n as u64 * value_dim as u64)));
    let layout = GdnTokenLayout { bhc: 0, c_len: 64, n_chunks: 0, heads: nvh, group, head_dim: khd, scale: 1.0f32 / (khd as f32).sqrt(), eps: 1e-6 };
    let conv = gdn_conv_silu_seg_step(
        g,
        &[mixed_qkv, w.conv1d_weight, &round.buf, pools.hist, &query, &key, &value],
        GdnConvLayout { rows: n, conv_dim, seq_len: n, key_dim, value_dim, kernel: shape.conv_kernel },
        table,
        bf16,
        pools.bf16,
        layout.eps,
    )?;

    // The prep writes every word of its slots it reads back, so the scratch is not cleared.
    let folds = gpu_core::provider::cuda::gdn_seg_loop_folds(g, bf16);
    // The folded recurrence's decay vectors take three words a row of a slot (see `gdn_chunk_prep_seg_bf16_step`).
    let (g_cs, exp_g_cs, u, wm, intra) = (g.storage_uninit(if folds { 3 * slots } else { slots }), g.storage_uninit(slots), g.storage_uninit(slots * vhd as u64), g.storage_uninit(slots * khd as u64), g.storage_uninit(slots * 64));
    // The scaled chunk-major copies the `mma.sync` recurrence reads; the warpgroup recurrence reads the convolution's rows instead.
    let (query_cm, key_cm) = if folds { (g.storage_uninit(1), g.storage_uninit(1)) } else { (g.storage_uninit(slots * khd as u64), g.storage_uninit(slots * khd as u64)) };
    let prep = gdn_chunk_prep_seg_bf16_step(g, &[&query, &key, &value, bproj, aproj, w.a_log, w.dt_bias, &round.buf, &g_cs, &exp_g_cs, &u, &wm, &intra, &query_cm, &key_cm], layout, table, bf16)?;
    // Every row of a segment is written by the recurrence (a mixed round's decode rows are the caller's).
    let out = g.storage_uninit(words(n as u64 * value_dim as u64));
    let (loop_k, loop_q) = if folds { (&key, &query) } else { (&key_cm, &query_cm) };
    let recurrence = gdn_chunk_loop_seg_bf16_step(g, &[loop_k, loop_q, &wm, &u, &g_cs, &intra, &round.buf, pools.state, &out], layout, table, bf16, pools.bf16)?;

    // Gated RMSNorm ("norm before gate"): RMSNorm(out) * weight, then * SiLU(z); one pass over the segments' rows.
    // (With `bf16` nothing stores it for the segments' rows; the buffer is the caller's to fill for the decode rows.)
    let gated = g.storage_uninit(n as u64 * value_dim as u64);
    // With `e4m3` the same pass writes the output projection's E4M3 activation (the rows `lead ..` of it).
    let (q, s) = if e4m3 { (g.storage_uninit(n as u64 * value_dim as u64 / 4), g.storage_uninit(n as u64 * nvh as u64)) } else { (g.storage_uninit(1), g.storage_uninit(1)) };
    let norm = gdn_gated_norm_step(g, &[&out, w.norm_weight, z, &gated, &q, &s], n * nvh, lead * nvh, shape.rms_eps, e4m3, bf16)?;

    g.submit(&[], &[conv, prep, recurrence, norm]);
    Some(SegmentsOut { gated, e4m3: e4m3.then_some((q, s)) })
}

#[cfg(target_arch = "wasm32")]
#[allow(clippy::too_many_arguments)]
pub fn gdn_mixer_segments_fwd(
    _g: &Gpu,
    _ids: &GdnMixerIds,
    _shape: &GdnMixerShape,
    _w: &GdnMixerWeights,
    _round: &GdnSegmentRound,
    _pools: &GdnSegmentPools,
    _mixed_qkv: &DeviceBuffer,
    _bproj: &DeviceBuffer,
    _aproj: &DeviceBuffer,
    _z: &DeviceBuffer,
    _n: u32,
    _lead: u32,
    _e4m3: bool,
    _bf16: bool,
) -> Option<SegmentsOut> {
    None
}

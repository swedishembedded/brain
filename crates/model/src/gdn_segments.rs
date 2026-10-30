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

/// The mixer of one layer over every segment of `round`, from the layer's projections `mixed_qkv`, `bproj`, `aproj`, `z`
/// (all `[n, ...]` for the round's `n` rows) to `gated` (`[n, value_dim]`). Rows not in a segment (a mixed round's
/// leading decode rows) come back undefined; the caller overwrites them. `None` (nothing submitted, the pools untouched)
/// when the device or the shape is not served.
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
) -> Option<DeviceBuffer> {
    use gpu_core::provider::cuda::{gdn_chunk_loop_seg_bf16_step, gdn_chunk_prep_seg_bf16_step, gdn_conv_silu_seg_step, GdnConvLayout, GdnTokenLayout};

    let gdn = shape.gdn;
    let (nvh, khd, vhd) = (gdn.h, gdn.dk, gdn.dv);
    let (conv_dim, key_dim, value_dim, group) = (shape.conv_dim(), shape.key_dim(), shape.value_dim(), shape.group());
    let chunk_shape = GdnShape { b: 1, h: nvh, t: n, dk: khd, dv: vhd, chunk: 64 };
    if !tokens_prefix_offered(g, &ids.chunk, &chunk_shape, group) || shape.conv_kernel != 4 {
        return None;
    }
    let table = &round.table;
    let slots = (table.chunks() * nvh) as u64 * 64;
    let (query, key, value) = (g.storage_uninit(n as u64 * key_dim as u64), g.storage_uninit(n as u64 * key_dim as u64), g.storage_uninit(n as u64 * value_dim as u64));
    let conv = gdn_conv_silu_seg_step(
        g,
        &[mixed_qkv, w.conv1d_weight, &round.buf, pools.hist, &query, &key, &value],
        GdnConvLayout { rows: n, conv_dim, seq_len: n, key_dim, value_dim, kernel: shape.conv_kernel },
        table,
    )?;

    let beta = g.storage_uninit(n as u64 * nvh as u64);
    let g_decay = g.storage_uninit(n as u64 * nvh as u64);
    // The prep writes every word of its slots it reads back, so the scratch is not cleared.
    let (g_cs, exp_g_cs, u, wm, intra) = (g.storage_uninit(slots), g.storage_uninit(slots), g.storage_uninit(slots * vhd as u64), g.storage_uninit(slots * khd as u64), g.storage_uninit(slots * 64));
    let (query_cm, key_cm) = (g.storage_uninit(slots * khd as u64), g.storage_uninit(slots * khd as u64));
    let layout = GdnTokenLayout { bhc: 0, c_len: 64, n_chunks: 0, heads: nvh, group, head_dim: khd, scale: 1.0f32 / (khd as f32).sqrt(), eps: 1e-6 };
    let prep = gdn_chunk_prep_seg_bf16_step(g, &[&query, &key, &value, &g_decay, &beta, &round.buf, &g_cs, &exp_g_cs, &u, &wm, &intra, &query_cm, &key_cm], layout, table)?;
    // Every row of a segment is written by the recurrence (a mixed round's decode rows are the caller's).
    let out = g.storage_uninit(n as u64 * value_dim as u64);
    let recurrence = gdn_chunk_loop_seg_bf16_step(g, &[&key_cm, &query_cm, &wm, &u, &g_cs, &intra, &round.buf, pools.state, &out], nvh, khd, vhd, table)?;

    g.submit(&[], &[g.step(ids.sigmoid, &[bproj, &beta], &[n * nvh], n * nvh), g.step(ids.gdn_decay_gate, &[aproj, w.a_log, w.dt_bias, &g_decay], &[n, nvh], n * nvh)]);
    g.submit(&[], &[conv, prep, recurrence]);

    // Gated RMSNorm ("norm before gate"): RMSNorm(out) * weight, then * SiLU(z); one pass over the round's rows.
    let normed = g.storage_uninit(n as u64 * value_dim as u64);
    let z_silu = g.storage_uninit(n as u64 * value_dim as u64);
    let gated = g.storage_uninit(n as u64 * value_dim as u64);
    g.submit(
        &[],
        &[
            crate::block::rmsnorm_fwd(g, &ids.kernels, &out, w.norm_weight, &normed, vhd, n * nvh, shape.rms_eps),
            g.step(ids.silu, &[z, &z_silu], &[n * value_dim], n * value_dim),
            g.step(ids.mul, &[&normed, &z_silu, &gated], &[n * value_dim], n * value_dim),
        ],
    );
    Some(gated)
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
) -> Option<DeviceBuffer> {
    None
}

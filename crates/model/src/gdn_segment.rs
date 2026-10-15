// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A Gated DeltaNet layer over a long sequence, run in SEGMENTS so the layer's
//! activation memory scales with the segment length instead of the sequence
//! length.
//!
//! Swedish Embedded AB implements solutions for training long-context
//! linear-attention models on a single accelerator for its clients. If your
//! team needs expertise in memory-bounded sequence-model training you can
//! procure our services by sending an email to info@swedishembedded.com.
//!
//! ## Why this exists
//!
//! [`crate::gdn_mixer::gdn_mixer_fwd`] with `is_train` keeps every
//! intermediate of the whole sequence alive until [`crate::gdn_mixer::
//! gdn_mixer_bwd`] runs: the per-token footprint is hundreds of kilobytes per
//! layer at the real Qwen3.5 width, so the sequence length a card can train is
//! set by one layer's saved set, even when the caller recomputes every layer.
//! A GDN layer is a recurrence, though, so a sequence can be cut at any chunk
//! boundary as long as the recurrent state and the causal conv's `K-1` row
//! history cross the cut. That is exactly what [`crate::gdn_mixer::GdnStream`]
//! already threads between prefill rounds.
//!
//! ## Contract
//!
//! * [`gdn_mixer_seg_fwd`] runs the segments in order with the cheap
//!   non-training forward and keeps only a snapshot of the stream (state +
//!   conv history, a few megabytes) at the start of each segment
//!   ([`GdnSegmentBounds`]). It returns the whole sequence's `gated`.
//! * [`gdn_mixer_seg_bwd`] walks the segments in REVERSE. For each one it
//!   re-runs the training forward from that segment's snapshot (so only one
//!   segment's activations exist at a time), runs
//!   [`crate::gdn_mixer::gdn_mixer_stream_bwd`] with the state gradient the
//!   following segment produced, and scatters the segment's input gradients
//!   into the whole-sequence result. The conv history's gradient is added to
//!   the last `K-1` rows of the previous segment's range. Weight gradients
//!   accumulate across segments in the caller's buffers.
//!
//! The segment length is rounded down to a multiple of the layer's chunk so
//! every segment is a whole number of chunks (the last one may be shorter),
//! and must be at least `conv_kernel-1` rows so a segment's conv history is
//! entirely inside the previous segment. The whole sequence as one segment
//! (`segment_tokens` 0 or `>= t`) is allowed and costs one recompute of the
//! layer forward on top of the one-shot pair.
//!
//! Requires `shape.gdn.b == 1`: segmenting cuts the token axis of a
//! `[T, C]` row-major buffer, which is contiguous only without a batch axis.

use gpu_core::{DeviceBuffer, Gpu};

use crate::gdn::GdnShape;
use crate::gdn_mixer::{gdn_mixer_stream_bwd, gdn_mixer_stream_fwd, GdnMixerGrads, GdnMixerIds, GdnMixerShape, GdnMixerWeights, GdnStream};

/// The four projections a GDN layer consumes, token-major `[t, width]`
/// (what [`crate::gdn_mixer::gdn_mixer_fwd`] takes as separate arguments).
#[derive(Clone, Copy)]
pub struct GdnMixerInputs<'a> {
    pub mixed_qkv: &'a DeviceBuffer,
    pub bproj: &'a DeviceBuffer,
    pub aproj: &'a DeviceBuffer,
    pub z: &'a DeviceBuffer,
}

/// What [`gdn_mixer_seg_fwd`] leaves for [`gdn_mixer_seg_bwd`]: the stream
/// snapshot at the start of every segment.
pub struct GdnSegmentBounds {
    spans: Vec<(u32, u32)>,
    /// `[bh, dk, dv]` recurrent state entering each segment.
    states: Vec<DeviceBuffer>,
    /// `[conv_kernel-1, conv_dim]` channel-major conv history entering each
    /// segment ([`GdnStream::hist`] layout).
    hists: Vec<DeviceBuffer>,
}

impl GdnSegmentBounds {
    pub fn n_segments(&self) -> usize {
        self.spans.len()
    }
}

/// The `(first_row, rows)` of every segment for a sequence of `t` rows cut
/// into chunk multiples of at most `segment_tokens` (0 = one segment).
pub fn segment_spans(t: u32, chunk: u32, segment_tokens: u32) -> Vec<(u32, u32)> {
    assert!(chunk > 0 && t.is_multiple_of(chunk), "segment_spans: t={t} must be a multiple of chunk={chunk}");
    let seg = if segment_tokens == 0 || segment_tokens >= t { t } else { (segment_tokens / chunk * chunk).max(chunk) };
    (0..t).step_by(seg.max(1) as usize).map(|row0| (row0, seg.min(t - row0))).collect()
}

fn checked_plan(shape: &GdnMixerShape, segment_tokens: u32) -> Vec<(u32, u32)> {
    assert_eq!(shape.gdn.b, 1, "gdn segmentation cuts the token axis, so b must be 1 (got {})", shape.gdn.b);
    let spans = segment_spans(shape.gdn.t, shape.gdn.chunk, segment_tokens);
    let history = shape.conv_kernel - 1;
    if spans.len() > 1 {
        assert!(
            spans[0].1 >= history,
            "gdn segmentation: segment of {} rows is shorter than the conv history ({history} rows); use a longer segment_tokens",
            spans[0].1
        );
    }
    spans
}

/// `rows * width` as a kernel parameter; a projection buffer beyond u32
/// elements cannot be addressed by the row kernels at all.
fn elems(rows: u32, width: u32) -> u32 {
    u32::try_from(rows as u64 * width as u64).unwrap_or_else(|_| panic!("gdn segmentation: {rows} rows x {width} exceeds the u32 element index of the row kernels"))
}

/// Rows `[row0, row0+rows)` of a `[t, width]` buffer as their own buffer (a
/// handle clone when that is the whole buffer).
fn take_rows(g: &Gpu, ids: &GdnMixerIds, src: &DeviceBuffer, width: u32, t: u32, (row0, rows): (u32, u32)) -> DeviceBuffer {
    if rows == t {
        return src.clone();
    }
    let dst = g.storage(rows as u64 * width as u64);
    let (total, count, off) = (elems(t, width), elems(rows, width), elems(row0, width));
    g.submit(&[], &[g.step(ids.concat_split, &[src, &dst], &[1, total, count, off, 1, 1], count)]);
    dst
}

/// `dst[row0*width ..] += src` for a `[rows, width]` source.
fn put_rows(g: &Gpu, ids: &GdnMixerIds, src: &DeviceBuffer, dst: &DeviceBuffer, width: u32, row0: u32, rows: u32) {
    let count = elems(rows, width);
    g.submit(&[], &[g.step(ids.chunk_bwd.splice_add, &[src, dst], &[count, elems(row0, width)], count)]);
}

fn copy(g: &Gpu, ids: &GdnMixerIds, src: &DeviceBuffer, len: u64) -> DeviceBuffer {
    let dst = g.storage(len);
    let len = u32::try_from(len).expect("stream snapshot exceeds u32 elements");
    g.submit(&[], &[g.step(ids.chunk.region_copy, &[src, &dst], &[1, len, len, 0], len)]);
    dst
}

fn segment_shape(shape: &GdnMixerShape, rows: u32) -> GdnMixerShape {
    GdnMixerShape { gdn: GdnShape { t: rows, ..shape.gdn }, ..*shape }
}

fn segment_inputs(g: &Gpu, ids: &GdnMixerIds, shape: &GdnMixerShape, inp: &GdnMixerInputs, span: (u32, u32)) -> [DeviceBuffer; 4] {
    let t = shape.gdn.t;
    [
        take_rows(g, ids, inp.mixed_qkv, shape.conv_dim(), t, span),
        take_rows(g, ids, inp.bproj, shape.gdn.h, t, span),
        take_rows(g, ids, inp.aproj, shape.gdn.h, t, span),
        take_rows(g, ids, inp.z, shape.value_dim(), t, span),
    ]
}

/// Forward over segments; see the module doc. `segment_tokens == 0` (or
/// `>= t`) runs one segment.
pub fn gdn_mixer_seg_fwd(g: &Gpu, ids: &GdnMixerIds, shape: &GdnMixerShape, w: &GdnMixerWeights, inp: &GdnMixerInputs, segment_tokens: u32) -> (DeviceBuffer, GdnSegmentBounds) {
    let spans = checked_plan(shape, segment_tokens);
    let (t, value_dim) = (shape.gdn.t, shape.value_dim());
    let state_len = shape.gdn.bh() as u64 * shape.gdn.dk as u64 * shape.gdn.dv as u64;
    let hist_len = shape.conv_dim() as u64 * (shape.conv_kernel - 1) as u64;

    // Zero-fresh: a fresh sequence starts from a zero state and zero history.
    let cur_state = g.storage(state_len);
    let cur_hist = g.storage(hist_len);
    let gated = if spans.len() > 1 { Some(g.storage(t as u64 * value_dim as u64)) } else { None };
    let mut bounds = GdnSegmentBounds { spans: spans.clone(), states: Vec::new(), hists: Vec::new() };
    let mut last = None;
    for &span in &spans {
        bounds.states.push(copy(g, ids, &cur_state, state_len));
        bounds.hists.push(copy(g, ids, &cur_hist, hist_len));
        let [mixed, b, a, z] = segment_inputs(g, ids, shape, inp, span);
        let seg = segment_shape(shape, span.1);
        let stream = GdnStream { state: &cur_state, hist: &cur_hist };
        let (out, _) = gdn_mixer_stream_fwd(g, ids, &seg, w, &mixed, &b, &a, &z, span.1, false, Some(stream));
        match &gated {
            Some(all) => put_rows(g, ids, &out, all, value_dim, span.0, span.1),
            None => last = Some(out),
        }
    }
    (gated.or(last).expect("at least one segment"), bounds)
}

/// Backward over segments in reverse, recomputing each segment's forward from
/// `bounds`. `inp` must be the same inputs [`gdn_mixer_seg_fwd`] saw.
/// Returns the whole-sequence `(d_mixed_qkv, d_bproj, d_aproj, d_z)`; weight
/// gradients accumulate into `gw`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_mixer_seg_bwd(
    g: &Gpu,
    ids: &GdnMixerIds,
    shape: &GdnMixerShape,
    w: &GdnMixerWeights,
    gw: &GdnMixerGrads,
    inp: &GdnMixerInputs,
    bounds: &GdnSegmentBounds,
    d_gated: &DeviceBuffer,
) -> (DeviceBuffer, DeviceBuffer, DeviceBuffer, DeviceBuffer) {
    let (t, nvh, conv_dim, value_dim) = (shape.gdn.t, shape.gdn.h, shape.conv_dim(), shape.value_dim());
    let history = shape.conv_kernel - 1;
    let state_len = shape.gdn.bh() as u64 * shape.gdn.dk as u64 * shape.gdn.dv as u64;
    let hist_len = conv_dim as u64 * history as u64;

    let d_mixed_qkv = g.storage(t as u64 * conv_dim as u64);
    let d_bproj = g.storage(t as u64 * nvh as u64);
    let d_aproj = g.storage(t as u64 * nvh as u64);
    let d_z = g.storage(t as u64 * value_dim as u64);

    let mut d_next_state: Option<DeviceBuffer> = None;
    for (s, &span) in bounds.spans.iter().enumerate().rev() {
        let (row0, rows) = span;
        let seg = segment_shape(shape, rows);
        let [mixed, b, a, z] = segment_inputs(g, ids, shape, inp, span);
        // The forward consumes (overwrites) its stream, so replay from copies.
        let state = copy(g, ids, &bounds.states[s], state_len);
        let hist = copy(g, ids, &bounds.hists[s], hist_len);
        let stream = GdnStream { state: &state, hist: &hist };
        let (_, acts) = gdn_mixer_stream_fwd(g, ids, &seg, w, &mixed, &b, &a, &z, rows, true, Some(stream));
        let acts = acts.expect("train forward saves activations");

        let d_gated_seg = take_rows(g, ids, d_gated, value_dim, t, span);
        let ((dm, db, da, dz), carry) = gdn_mixer_stream_bwd(g, ids, &seg, w, gw, &acts, &d_gated_seg, rows, d_next_state.as_ref());
        drop(acts);

        put_rows(g, ids, &dm, &d_mixed_qkv, conv_dim, row0, rows);
        put_rows(g, ids, &db, &d_bproj, nvh, row0, rows);
        put_rows(g, ids, &da, &d_aproj, nvh, row0, rows);
        put_rows(g, ids, &dz, &d_z, value_dim, row0, rows);
        let carry = carry.expect("a continued forward yields the stream gradient");
        if s > 0 {
            // This segment's conv history IS the previous segment's last
            // `history` input rows (checked_plan guarantees they exist).
            put_rows(g, ids, &carry.d_hist, &d_mixed_qkv, conv_dim, row0 - history, history);
        }
        d_next_state = Some(carry.d_state);
    }
    (d_mixed_qkv, d_bproj, d_aproj, d_z)
}

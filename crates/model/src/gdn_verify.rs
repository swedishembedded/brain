// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements speculative decoding - drafters, verifiers
// and the state rollback a recurrent stack needs - for clients serving large
// models on modest hardware. If your team needs expertise in verifying several
// tokens per sequence through a Gated DeltaNet layer without losing the ability
// to undo a rejection, then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The recurrent half of a speculative **verify** step over a batch.
//!
//! A decode step gives each sequence one token. A verify step gives sequence
//! `i` `counts[i]` consecutive ones (its next input and the proposals after
//! it), packed into one token array, and some of those will be rejected. The
//! Gated DeltaNet state absorbs every row it is shown, so a rejection needs
//! the state AFTER the last accepted row back - which the in-place update has
//! already moved past. [`gdn_mixer_verify_pooled_fused`] therefore keeps the
//! state and conv window after every row but the last (the last row's state
//! is the pool's, exactly as a plain step leaves it, so accepting everything
//! needs no repair), and [`gdn_commit_pooled`] copies a chosen snapshot back
//! over the pool row of each sequence that rejected something.
//!
//! A bf16 pool is verified by the streaming kernel the plain bf16 decode uses, run over the rows with the state rounded to
//! bf16 after each, so a verified row is bit for bit a plain step and a rejection restores an exact bf16 state.
//!
//! Native (CUDA) kernels only: a device without them returns `None` and the
//! decoder reports that it does not speculate.

use gpu_core::{DeviceBuffer, Gpu, Step};

use crate::gdn_mixer::{GdnMixerShape, GdnMixerWeights, GdnPoolRows};

/// Most rows one sequence may carry through a verify launch
/// (`BRAIN_GDN_MAX_ROWS` in `cu/gdn_decode.cu`).
pub const MAX_VERIFY_ROWS: u32 = 8;

/// The recurrent state and conv window after each verified row of one layer,
/// packed like the token array: sequence `i`'s row `r` is entry
/// `offs[i] + r`; the last row of each sequence is never written.
pub struct GdnRowSnapshots {
    /// `[n_tokens, state_len]`.
    pub state: DeviceBuffer,
    /// `[n_tokens, hist_len]`.
    pub hist: DeviceBuffer,
}

impl GdnRowSnapshots {
    /// Scratch for `n_tokens` packed verify rows of a layer shaped `shape`.
    pub fn new(g: &Gpu, shape: &GdnMixerShape, n_tokens: u32) -> GdnRowSnapshots {
        let state_len = shape.gdn.h as u64 * shape.gdn.dk as u64 * shape.gdn.dv as u64;
        let hist_len = shape.conv_dim() as u64 * (shape.conv_kernel as u64 - 1);
        GdnRowSnapshots { state: g.storage(n_tokens as u64 * state_len), hist: g.storage(n_tokens as u64 * hist_len) }
    }
}

/// How a verify launch finds its sequences: the pool row of each, and how many
/// packed token rows each owns.
pub struct VerifyLayout<'a> {
    /// `[b]` u32: the pool row of each sequence, distinct.
    pub rows: &'a DeviceBuffer,
    /// `[b]` u32: rows of each sequence, `1..=MAX_VERIFY_ROWS`.
    pub counts: &'a DeviceBuffer,
    /// `[b]` u32: index of each sequence's first row in the packed token arrays.
    pub offs: &'a DeviceBuffer,
    /// Sequences in the batch.
    pub b: u32,
}

/// The conv, gates, delta-rule update and gated norm of every verified row, in
/// one native launch, each sequence's rows taken in order from its pool row of
/// `pool` (updated in place to the state after its LAST row), snapshotting the
/// state and window after the rest into `snaps`. `mixed`, `bproj`, `aproj` and
/// `z` carry the packed token rows; the result is the packed `gated` block
/// (`[n_tokens, value_dim]`). Each row is what a plain decode step produces for
/// it, bit for bit.
///
/// A bf16 pool (`pool.bf16`) runs the streaming kernel (`gdn_verify_pool_stream_bf16`): the body of the plain bf16 decode over
/// a row loop, the state tile and conv window rounded to bf16 after EVERY row as a plain step rounds them, so each row is
/// bit for bit the plain step (`pool.wide` set, the serving default) and greedy speculation is token for token plain
/// decoding. The snapshots stay f32, the widening of those bf16 values, so a restore is exact.
///
/// `None`, nothing submitted, where the device is not offered the kernel or
/// the head shape is outside its block shape (128-wide key and value heads, a
/// 4-tap conv, at most three value heads per key head).
#[allow(clippy::too_many_arguments)]
pub fn gdn_mixer_verify_pooled_fused(
    g: &Gpu,
    shape: &GdnMixerShape,
    w: &GdnMixerWeights,
    mixed: &DeviceBuffer,
    bproj: &DeviceBuffer,
    aproj: &DeviceBuffer,
    z: &DeviceBuffer,
    pool: &GdnPoolRows,
    layout: &VerifyLayout,
    n_tokens: u32,
    snaps: &GdnRowSnapshots,
) -> Option<DeviceBuffer> {
    let gdn = shape.gdn;
    if gdn.dk != 128 || gdn.dv != 128 || shape.conv_kernel != 4 || shape.nkh == 0 || !gdn.h.is_multiple_of(shape.nkh) {
        return None;
    }
    let gated = g.storage(n_tokens as u64 * shape.value_dim() as u64);
    let f = f32::to_bits;
    let params = [shape.nkh, gdn.h, shape.group(), f(1e-6), f(shape.rms_eps), f(1.0f32 / (gdn.dk as f32).sqrt()), layout.b, 0];
    let fused = if pool.bf16 { gpu_core::Fused::GdnVerifyPoolStreamBf16 } else { gpu_core::Fused::GdnVerifyPool };
    let bufs = [mixed, w.conv1d_weight, pool.hist, bproj, aproj, w.a_log, w.dt_bias, pool.state, z, w.norm_weight, &gated, layout.rows, layout.counts, layout.offs, &snaps.state, &snaps.hist];
    if pool.bf16 {
        let step = g.fused_step(fused, &bufs, &params)?;
        g.submit(&[], &[step]);
        return Some(gated);
    }
    // The split form first: a conv thread per channel and a recurrence block per value head, bit-identical to the fused body
    // below, whose one block per key head spills its registers and fills 16 of the card's 132 SMs at one sequence.
    let (conv, recur) = (gpu_core::Fused::GdnConvVerifyPool, gpu_core::Fused::GdnRecurVerifyPool);
    if g.has_fused(conv) && g.has_fused(recur) && conv.serves(&[shape.nkh, gdn.h, layout.b]) && recur.serves(&params) {
        let conv_out = g.storage_uninit(n_tokens as u64 * shape.conv_dim() as u64);
        let conv_step = g.fused_step(conv, &[mixed, w.conv1d_weight, pool.hist, &conv_out, layout.rows, layout.counts, layout.offs, &snaps.hist], &[shape.nkh, gdn.h, layout.b]);
        let recur_step = conv_step
            .is_some()
            .then(|| g.fused_step(recur, &[&conv_out, bproj, aproj, w.a_log, w.dt_bias, pool.state, z, w.norm_weight, &gated, layout.rows, layout.counts, layout.offs, &snaps.state], &params))
            .flatten();
        if let (Some(conv_step), Some(recur_step)) = (conv_step, recur_step) {
            g.submit(&[], &[conv_step, recur_step]);
            return Some(gated);
        }
    }
    let step = g.fused_step(fused, &bufs, &params)?;
    g.submit(&[], &[step]);
    Some(gated)
}

/// [`gdn_mixer_verify_pooled_fused`] over the rows of one fused projection `qkvz` (`[n_tokens, qkvz_row_width]`, a row's qkv then
/// its z), read in place by the streaming kernel: the bits of the launch over two separate arrays holding the same values. `None`
/// - nothing submitted - where [`crate::gdn_mixer::qkvz_in_place_offered`] says no.
#[allow(clippy::too_many_arguments)]
pub fn gdn_mixer_verify_pooled_qkvz(
    g: &Gpu,
    shape: &GdnMixerShape,
    w: &GdnMixerWeights,
    qkvz: &DeviceBuffer,
    bproj: &DeviceBuffer,
    aproj: &DeviceBuffer,
    pool: &GdnPoolRows,
    layout: &VerifyLayout,
    n_tokens: u32,
    snaps: &GdnRowSnapshots,
) -> Option<DeviceBuffer> {
    if !crate::gdn_mixer::qkvz_in_place_offered(g, shape, pool) || !g.has_fused(gpu_core::Fused::GdnVerifyPoolStreamBf16) {
        return None;
    }
    let gdn = shape.gdn;
    let gated = g.storage(n_tokens as u64 * shape.value_dim() as u64);
    let f = f32::to_bits;
    let params = [shape.nkh, gdn.h, shape.group(), f(1e-6), f(shape.rms_eps), f(1.0f32 / (gdn.dk as f32).sqrt()), layout.b, crate::gdn_mixer::qkvz_row_width(shape)];
    let bufs = [qkvz, w.conv1d_weight, pool.hist, bproj, aproj, w.a_log, w.dt_bias, pool.state, qkvz, w.norm_weight, &gated, layout.rows, layout.counts, layout.offs, &snaps.state, &snaps.hist];
    let step = g.fused_step(gpu_core::Fused::GdnVerifyPoolStreamBf16, &bufs, &params)?;
    g.submit(&[], &[step]);
    Some(gated)
}

/// Put snapshot row `sel[i]` back over pool row `rows[i]` for every sequence
/// whose `sel` entry is not `u32::MAX` - the repair for a verify that rejected
/// part of a sequence's rows. A bf16 pool (`bf16`) takes the f32 snapshot
/// narrowed back, which is exact: the snapshot is the widening of its bf16
/// values. The step is returned, not submitted, so a caller can batch every
/// layer's repair into one submission.
///
/// `None` where the device is not offered the kernel.
#[allow(clippy::too_many_arguments)]
pub fn gdn_commit_pooled(g: &Gpu, shape: &GdnMixerShape, pool_state: &DeviceBuffer, pool_hist: &DeviceBuffer, bf16: bool, snaps: &GdnRowSnapshots, rows: &DeviceBuffer, sel: &DeviceBuffer, b: u32) -> Option<Step> {
    let state_len = shape.gdn.h * shape.gdn.dk * shape.gdn.dv;
    let hist_len = shape.conv_dim() * (shape.conv_kernel - 1);
    let fused = if bf16 { gpu_core::Fused::GdnCommitPoolBf16 } else { gpu_core::Fused::GdnCommitPool };
    g.fused_step(fused, &[pool_state, pool_hist, &snaps.state, &snaps.hist, rows, sel], &[b, state_len, hist_len, 0])
}

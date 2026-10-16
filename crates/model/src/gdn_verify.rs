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
    let step = g.fused_step(
        gpu_core::Fused::GdnVerifyPool,
        &[mixed, w.conv1d_weight, pool.hist, bproj, aproj, w.a_log, w.dt_bias, pool.state, z, w.norm_weight, &gated, layout.rows, layout.counts, layout.offs, &snaps.state, &snaps.hist],
        &params,
    )?;
    g.submit(&[], &[step]);
    Some(gated)
}

/// Put snapshot row `sel[i]` back over pool row `rows[i]` for every sequence
/// whose `sel` entry is not `u32::MAX` - the repair for a verify that rejected
/// part of a sequence's rows. The step is returned, not submitted, so a caller
/// can batch every layer's repair into one submission.
///
/// `None` where the device is not offered the kernel.
#[allow(clippy::too_many_arguments)]
pub fn gdn_commit_pooled(g: &Gpu, shape: &GdnMixerShape, pool_state: &DeviceBuffer, pool_hist: &DeviceBuffer, snaps: &GdnRowSnapshots, rows: &DeviceBuffer, sel: &DeviceBuffer, b: u32) -> Option<Step> {
    let state_len = shape.gdn.h * shape.gdn.dk * shape.gdn.dv;
    let hist_len = shape.conv_dim() * (shape.conv_kernel - 1);
    g.fused_step(gpu_core::Fused::GdnCommitPool, &[pool_state, pool_hist, &snaps.state, &snaps.hist, rows, sel], &[b, state_len, hist_len, 0])
}

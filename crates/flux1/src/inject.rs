// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Per-block injection points on the FLUX.1 forward — the seam an *adapter*
//! (PuLID identity conditioning, a FLUX ControlNet, an IP-Adapter) uses to add
//! its own contribution to the residual stream between backbone blocks.
//!
//! Why a seam rather than an adapter-shaped field on [`crate::Flux1Model`]:
//! every one of those adapters is "run some extra dispatches on the image rows
//! after block *i*", they differ only in *which* dispatches, and the model has
//! no business knowing about any of them. `Flux1Model` therefore exposes the
//! two facts an adapter needs — the residual slab and where the image rows
//! start in it — and takes back a list of steps.
//!
//! The steps an implementor pushes MUST be built from the SAME [`gpu_core::Gpu`]
//! handle the model was built with: a `Step` carries a pipeline index into the
//! kernel list that handle was constructed from. The intended pattern is that
//! the adapter's kernels are appended to the model's list (see
//! `pulid::model::joint_kernels`), so one handle serves both and the whole
//! conditioned forward stays a single submit.
//!
//! In-place mutation of `x` is expected and correct: the FLUX.1 forward is
//! inference-only, so there is no SSA activation cache to preserve. An adapter
//! used in a *training-mode* forward would have to write a fresh buffer.

use gpu_core::{DeviceBuffer, Step};

/// The residual slab at one injection point.
///
/// Layout is FLUX.1's joint `[n, d]` slab with **text rows first**: rows
/// `0..n_txt` are text, rows `n_txt..n` are image (plus, on the Kontext edit
/// path, the appended reference-image tokens).
#[derive(Clone, Copy)]
pub struct InjectSite<'a> {
    /// The joint residual slab, `[n, d]` row-major, live and writable.
    pub x: &'a DeviceBuffer,
    /// First image row.
    pub n_txt: u32,
    /// Total rows (`n_txt + image + reference`).
    pub n: u32,
    /// Row width (`Flux1Config::hidden`).
    pub d: u32,
    /// Rows of the **noise span** — the leading `n_pred` image rows the final
    /// layer actually predicts. On a text-to-image run this equals
    /// [`Self::n_img`]; on the Kontext edit path the remaining
    /// `n_img() - n_pred` rows are the appended, *conditioning* reference-image
    /// tokens. An adapter that must not touch the reference tokens operates on
    /// `n_txt .. n_txt + n_pred` instead of `n_txt .. n`; the distinction is
    /// invisible from `n_txt`/`n` alone, which is why it is a field and not a
    /// caller's assumption.
    pub n_pred: u32,
}

impl InjectSite<'_> {
    /// Image (and reference) row count — the whole image stream.
    pub fn n_img(&self) -> u32 {
        self.n - self.n_txt
    }

    /// Row range of the noise span, `(first row, row count)`.
    pub fn pred_rows(&self) -> (u32, u32) {
        (self.n_txt, self.n_pred)
    }
}

/// An adapter that contributes dispatches after a backbone block.
///
/// Both methods are called once per block, in dispatch order, with the block's
/// *output* already in `site.x`. An implementation that has nothing to do at
/// this block simply pushes nothing.
pub trait BlockInject {
    /// After double-stream block `bi`.
    fn after_double(&self, bi: usize, site: InjectSite<'_>, steps: &mut Vec<Step>);
    /// After single-stream block `bi`.
    fn after_single(&self, bi: usize, site: InjectSite<'_>, steps: &mut Vec<Step>);
}

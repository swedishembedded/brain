// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`PulidAdapter`] — [`PulidCa`] driven through `flux1`'s [`BlockInject`]
//! seam, so a conditioned FLUX.1 forward is one call and one submit.
//!
//! The schedule is [`PulidConfig::schedule`], computed once for the backbone's
//! depth; `after_double` / `after_single` look their block up in it and, if it
//! is a site, append that module's dispatches over the image rows
//! (`site.n_txt .. site.n`). A block that is not a site pushes nothing.
//!
//! **Which image rows are conditioned.** All of them — `site.n_txt .. site.n`.
//! The reference adds the contribution to `img`, the whole image stream, and on
//! a text-to-image run (which is the only run PuLID-FLUX has: it is built on
//! FLUX.1-dev, not Kontext) that IS the whole span. brain's backbone is
//! Kontext-shaped, so on an *edit* run the image stream also carries the
//! appended reference-image tokens, and this adapter conditions those too. That
//! is a choice with no reference to check it against; an adapter that wanted the
//! noise span only would use `site.pred_rows()`. Nothing in this crate's parity
//! ladder exercises the edit path (`n_pred == n_img()` in every golden).
//!
//! **`id_weight` and the start step.** `id_weight` is read when the steps are
//! built, i.e. on every forward, so changing it is a field write, not a graph
//! rebuild. The reference's `start_step` is *not* modelled here because it is a
//! sampler-loop property: `flux/sampling.py` passes `id=None` for steps below
//! it, which in brain is the plain `Flux1Model::forward`. `crates/flux1` has no
//! sampler loop, so nothing here could schedule against one.

use std::cell::Cell;

use flux1::inject::{BlockInject, InjectSite};
use gpu_core::Step;

use crate::config::{PulidConfig, Site, Stream};
use crate::model::PulidCa;

pub struct PulidAdapter {
    ca: PulidCa,
    schedule: Vec<Site>,
    id_weight: Cell<f32>,
}

impl PulidAdapter {
    /// `depth_double` / `depth_single` are the backbone's block counts — the
    /// reduced-depth parity gate builds a shorter one, and the schedule follows
    /// the same sequential `ca_idx` counter the reference uses.
    pub fn new(ca: PulidCa, cfg: &PulidConfig, depth_double: usize, depth_single: usize, id_weight: f32) -> PulidAdapter {
        let schedule = cfg.schedule(depth_double, depth_single);
        assert!(
            schedule.iter().all(|s| s.ca < ca.n_ca()),
            "pulid: backbone depth {depth_double}+{depth_single} needs {} cross-attention modules, checkpoint has {}",
            schedule.len(),
            ca.n_ca()
        );
        PulidAdapter { ca, schedule, id_weight: Cell::new(id_weight) }
    }

    /// Upload the [`crate::IdFormer`] output.
    pub fn set_id(&self, id: &[f32]) {
        self.ca.set_id(id);
    }

    /// The identity strength dial. Read at step-build time, so a change takes
    /// effect on the next forward with no rebuild.
    pub fn set_id_weight(&self, w: f32) {
        self.id_weight.set(w);
    }
    pub fn id_weight(&self) -> f32 {
        self.id_weight.get()
    }

    /// The cross-attention sites for the depth this adapter was built for.
    pub fn schedule(&self) -> &[Site] {
        &self.schedule
    }

    fn fire(&self, stream: Stream, bi: usize, site: InjectSite<'_>, steps: &mut Vec<Step>) {
        let Some(s) = self.schedule.iter().find(|s| s.stream == stream && s.block == bi) else {
            return;
        };
        self.ca.inject_steps(
            steps,
            s.ca,
            site.x,
            site.n_txt as usize,
            site.n_img() as usize,
            self.id_weight.get(),
        );
    }
}

impl BlockInject for PulidAdapter {
    fn after_double(&self, bi: usize, site: InjectSite<'_>, steps: &mut Vec<Step>) {
        self.fire(Stream::Double, bi, site, steps);
    }
    fn after_single(&self, bi: usize, site: InjectSite<'_>, steps: &mut Vec<Step>) {
        self.fire(Stream::Single, bi, site, steps);
    }
}

// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The searches as [`ModelFit`] visitors, so a model family's trainer builds
//! the model and the search runs on whichever concrete [`Model`] it hands over.
//!
//! Both read a prepared chat dataset ([`data::chat::prepare_chat_samples`]):
//! its train split is the calibration set, its validation split the held-out
//! set, so the two are disjoint by construction.

use std::path::Path;

use gpu_core::select::Dtype;
use model::finetune::ModelFit;
use model::{Batch, FitOpts, FitReport, Model};

use crate::{held_out_loss, search_weights, Budget, Report, SearchRequest};

/// Every example row of one split of a prepared dataset, `opts.batch_size`
/// rows a batch, as owned token and target buffers.
fn rows_of(split: &data::loader::TokenDataset, cfg: &data::loader::BatchConfig) -> Vec<(Vec<u32>, Vec<u32>)> {
    let count = split.example_count().unwrap_or(0);
    let mut out = Vec::new();
    let mut first = 0;
    while first < count {
        let Some((x, y)) = split.example_rows(cfg, first) else { break };
        out.push((x, model::train::targets_to_u32(&y)));
        first += cfg.batch_size.max(1);
    }
    out
}

fn lm(owned: &[(Vec<u32>, Vec<u32>)]) -> Vec<Batch<'_>> {
    owned.iter().map(|(tokens, targets)| Batch::Lm { tokens, targets }).collect()
}

/// A report for a visit that trained nothing.
fn no_steps(loss: f64) -> FitReport {
    FitReport { initial_loss: loss as f32, final_loss: None, steps_completed: 0, resumed_at: None, interrupted: false, evaluations: Vec::new(), kept_best: None, stopped_early: false }
}

/// [`search_weights`] over the model a trainer builds, on the prepared
/// dataset at `dir`. The model needs gradients: build it trainable.
pub struct SearchFit<'a> {
    pub dir: &'a Path,
    pub opts: &'a FitOpts,
    pub tiers: &'a [Dtype],
    pub budget: Budget,
    /// The search's report, once the visit has run.
    pub report: Option<Report>,
}

impl ModelFit for SearchFit<'_> {
    fn fit<M: Model>(&mut self, model: M) -> std::io::Result<(FitReport, M)> {
        let (train, val, cfg, _) = model::load_dataset(self.dir, self.opts)?;
        let (calibration, held_out) = (rows_of(&train, &cfg), rows_of(&val, &cfg));
        let req = SearchRequest { tiers: self.tiers, budget: self.budget, calibration: &lm(&calibration), held_out: &lm(&held_out) };
        let report = search_weights(&model, &req).map_err(std::io::Error::other)?;
        let fit = no_steps(report.original_loss);
        self.report = Some(report);
        Ok((fit, model))
    }
}

/// The held-out loss of the model a trainer builds, on the validation split of
/// the prepared dataset at `dir`: how a plan is measured on its deployed
/// kernels once the trainer has built the model at it.
pub struct MeasureFit<'a> {
    pub dir: &'a Path,
    pub opts: &'a FitOpts,
    /// The measured loss, once the visit has run.
    pub loss: Option<f64>,
}

impl ModelFit for MeasureFit<'_> {
    fn fit<M: Model>(&mut self, model: M) -> std::io::Result<(FitReport, M)> {
        let (_, val, cfg, _) = model::load_dataset(self.dir, self.opts)?;
        let held_out = rows_of(&val, &cfg);
        if held_out.is_empty() {
            return Err(std::io::Error::other("the prepared dataset has no held-out rows to measure on"));
        }
        let loss = held_out_loss(&model, &lm(&held_out));
        self.loss = Some(loss);
        Ok((no_steps(loss), model))
    }
}

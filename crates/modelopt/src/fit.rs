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

use crate::{held_out_loss, search_weights, Budget, Progress, Report, SearchRequest, Timing};

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
    pub on_progress: &'a mut dyn FnMut(Progress),
    pub timing: Option<Timing<'a>>,
    pub frontier_points: usize,
    /// The search's report, once the visit has run.
    pub report: Option<Report>,
}

impl ModelFit for SearchFit<'_> {
    fn fit<M: Model>(&mut self, model: M) -> std::io::Result<(FitReport, M)> {
        let (train, val, cfg, _) = model::load_dataset(self.dir, self.opts)?;
        let (calibration, held_out) = (model::example_rows(&train, &cfg), model::example_rows(&val, &cfg));
        let req = SearchRequest { tiers: self.tiers, budget: self.budget, calibration: &lm(&calibration), held_out: &lm(&held_out), timing: self.timing.as_ref().map(|t| Timing { rows: t.rows, limit: t.limit, time: t.time }), frontier_points: self.frontier_points };
        let report = search_weights(&model, &req, self.on_progress).map_err(std::io::Error::other)?;
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
        let held_out = model::example_rows(&val, &cfg);
        if held_out.is_empty() {
            return Err(std::io::Error::other("the prepared dataset has no held-out rows to measure on"));
        }
        let loss = held_out_loss(&model, &lm(&held_out));
        self.loss = Some(loss);
        Ok((no_steps(loss), model))
    }
}

/// How narrow a pruned dimension becomes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PruneWidth {
    /// This share of each dimension's channels, rounded to the nearest channel.
    Fraction(f64),
    Channels(usize),
}

impl PruneWidth {
    fn channels(self, width: usize) -> Result<usize, String> {
        match self {
            PruneWidth::Channels(n) => Ok(n),
            PruneWidth::Fraction(f) if f > 0.0 && f <= 1.0 => Ok(((width as f64 * f).round() as usize).max(1)),
            PruneWidth::Fraction(f) => Err(format!("a kept fraction must be in (0, 1], got {f}")),
        }
    }
}

/// What a pruning run decided and measured on the trainable build.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct PruneReport {
    pub config_key: &'static str,
    pub width_before: usize,
    pub width_after: usize,
    pub params_before: usize,
    pub params_after: usize,
    pub original_loss: f64,
    /// The held-out loss with the dropped channels zeroed: what the narrower model computes.
    pub pruned_loss: f64,
    /// Per pruned dimension: its name and the channels kept.
    pub kept: Vec<(String, Vec<usize>)>,
}

/// Structured pruning over the model a trainer builds (trainable, for the
/// gradients the ranking needs), on the prepared dataset at `dir`: rank the
/// channels of [`Model::prunable_dims`] on the calibration split, keep `width`
/// of each, measure the original and the plan on the held-out split, and save
/// the narrower model to `out` through the family's own [`Model::save`].
pub struct PruneFit<'a> {
    pub dir: &'a Path,
    pub opts: &'a FitOpts,
    pub width: PruneWidth,
    pub out: &'a Path,
    pub on_progress: &'a mut dyn FnMut(Progress),
    /// The report, once the visit has run.
    pub report: Option<PruneReport>,
}

impl ModelFit for PruneFit<'_> {
    fn fit<M: Model>(&mut self, model: M) -> std::io::Result<(FitReport, M)> {
        let (train, val, cfg, _) = model::load_dataset(self.dir, self.opts)?;
        let (calibration, held_out) = (model::example_rows(&train, &cfg), model::example_rows(&val, &cfg));
        let dims = model.prunable_dims();
        let first = dims.first().ok_or_else(|| std::io::Error::other("the model declares no prunable dimension"))?;
        let (config_key, width_before) = (first.config_key, first.width);
        let width = self.width.channels(width_before).map_err(std::io::Error::other)?;
        let importance = crate::channel_importance(&model, &dims, &lm(&calibration), self.on_progress).map_err(std::io::Error::other)?;
        let plan = crate::plan_prune(&dims, &importance, width).map_err(std::io::Error::other)?;
        (self.on_progress)(Progress::Measuring { label: "original".into() });
        let original_loss = held_out_loss(&model, &lm(&held_out));
        (self.on_progress)(Progress::Measuring { label: "pruned".into() });
        let pruned_loss = crate::measure_pruned(&model, &dims, &plan, &lm(&held_out)).map_err(std::io::Error::other)?;
        let narrow = crate::pruned_weights(&model, &dims, &plan).map_err(std::io::Error::other)?;
        let params_before = model.param_names().iter().map(|n| model.read_weight(n).len()).sum();
        let params_after = narrow.weights.values().map(Vec::len).sum();
        let out = self.out.to_str().ok_or_else(|| std::io::Error::other(format!("{}: not a UTF-8 path", self.out.display())))?;
        M::new_inference(narrow.config, 1, self.opts.block_size, &narrow.weights).save(out);
        let kept = dims.iter().zip(plan.keep).map(|(d, k)| (d.name.clone(), k)).collect();
        self.report = Some(PruneReport { config_key, width_before, width_after: width, params_before, params_after, original_loss, pruned_loss, kept });
        Ok((no_steps(original_loss), model))
    }
}

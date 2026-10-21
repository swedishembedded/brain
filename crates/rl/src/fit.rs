// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for aligning and distilling
// open-weight language models of any architecture through one training stack,
// for its clients. If your team needs expertise in preference, reward-driven
// and distillation fine-tuning of quantized models then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The objectives of this crate as [`ModelFit`] visitors.
//!
//! A model family implements `model::finetune::ChatTrainer` once; it builds its
//! model (over any frozen-base encoding it supports) and hands it to a
//! [`ModelFit`]. The visitors here are the post-training objectives: each runs
//! [`model::fit_controlled`] with one of this crate's [`Objective`]s over
//! whichever concrete [`Model`] it is given, so DPO, GRPO and distillation are
//! written once and run on every family and every base encoding the family's
//! trainer offers. None of them names an architecture.
//!
//! * [`DpoFit`] - preference pairs against the starting policy as reference.
//! * [`GrpoFit`] - group-relative policy optimisation against an environment.
//! * [`DistillFit`] - top-K forward-KL distillation from teacher distributions.
//!   Every pass is a weighted cross-entropy, which `Model` computes through the
//!   chunked LM head (`model::head_ce`), so the logits of a whole vocabulary
//!   are never held for a whole row.

use model::finetune::ModelFit;
use model::{fit_controlled, FitControl, FitOpts, FitReport, Model};

use crate::env::{Environment, Verifier};
use crate::objective::distill::{DistillConfig, DistillExample, DistillTopK};
use crate::objective::dpo::{Dpo, DpoConfig, PackedPair};
use crate::objective::grpo::Grpo;

/// The control a visitor consumes: a visitor trains one model.
fn take_control<'c>(control: &mut Option<FitControl<'c>>, what: &str) -> std::io::Result<FitControl<'c>> {
    control.take().ok_or_else(|| std::io::Error::other(format!("a {what} fit trains one model")))
}

/// DPO over whichever model the trainer builds: the objective is made per
/// concrete type from the packed pairs the visitor holds, and the pair sets to
/// score are measured on that model before and after.
pub struct DpoFit<'a, 'c> {
    dpo: DpoConfig,
    pairs: Option<Vec<PackedPair>>,
    score_sets: Vec<Vec<PackedPair>>,
    margins: Vec<Vec<f64>>,
    opts: &'a FitOpts,
    control: Option<FitControl<'c>>,
}

impl<'a, 'c> DpoFit<'a, 'c> {
    /// `score_sets` are measured on the resident model before the first step
    /// (their reference is the starting policy) and after the last.
    pub fn new(dpo: DpoConfig, pairs: Vec<PackedPair>, score_sets: Vec<Vec<PackedPair>>, opts: &'a FitOpts, control: FitControl<'c>) -> DpoFit<'a, 'c> {
        DpoFit { dpo, pairs: Some(pairs), score_sets, margins: Vec::new(), opts, control: Some(control) }
    }

    /// One margin list per score set, measured after an uninterrupted run
    /// (empty otherwise).
    pub fn margins(&self) -> &[Vec<f64>] {
        &self.margins
    }

    /// [`Self::margins`], consuming the visitor.
    pub fn into_margins(self) -> Vec<Vec<f64>> {
        self.margins
    }
}

impl ModelFit for DpoFit<'_, '_> {
    fn fit<M: Model>(&mut self, model: M) -> std::io::Result<(FitReport, M)> {
        let pairs = self.pairs.take().ok_or_else(|| std::io::Error::other("a preference fit trains one model"))?;
        let control = take_control(&mut self.control, "preference")?;
        // The starting policy is the reference, as it is for the training objective.
        for set in &mut self.score_sets {
            set.iter_mut().for_each(|pair| pair.score_reference(&model));
        }
        let (report, model) = fit_controlled(model, Dpo::from_dataset(self.dpo, pairs), self.opts, None, control)?;
        if !report.interrupted {
            self.margins = self.score_sets.iter().map(|set| set.iter().map(|pair| pair.margin(&model)).collect()).collect();
        }
        Ok((report, model))
    }
}

/// GRPO over whichever model the trainer builds. The objective is built by the
/// caller (environment, verifier, configuration, optional reference and
/// rollout) and stays inspectable through [`Self::grpo`] after the run.
pub struct GrpoFit<'a, 'c, E: Environment, V: Verifier> {
    grpo: Grpo<E, V>,
    opts: &'a FitOpts,
    control: Option<FitControl<'c>>,
}

impl<'a, 'c, E: Environment, V: Verifier> GrpoFit<'a, 'c, E, V> {
    pub fn new(grpo: Grpo<E, V>, opts: &'a FitOpts, control: FitControl<'c>) -> GrpoFit<'a, 'c, E, V> {
        GrpoFit { grpo, opts, control: Some(control) }
    }

    /// The objective, for its statistics.
    pub fn grpo(&self) -> &Grpo<E, V> {
        &self.grpo
    }
}

impl<E: Environment, V: Verifier> ModelFit for GrpoFit<'_, '_, E, V> {
    fn fit<M: Model>(&mut self, model: M) -> std::io::Result<(FitReport, M)> {
        let control = take_control(&mut self.control, "GRPO")?;
        fit_controlled(model, &mut self.grpo, self.opts, None, control)
    }
}

/// Top-K distillation over whichever model the trainer builds: the student
/// learns the teacher's per-position distributions carried by `examples`.
pub struct DistillFit<'a, 'c> {
    distill: Option<DistillTopK>,
    opts: &'a FitOpts,
    control: Option<FitControl<'c>>,
}

impl<'a, 'c> DistillFit<'a, 'c> {
    /// `examples` are cycled; `cfg.seq_len` must be the row length the model is built for.
    pub fn new(cfg: DistillConfig, examples: Vec<DistillExample>, opts: &'a FitOpts, control: FitControl<'c>) -> DistillFit<'a, 'c> {
        DistillFit { distill: Some(DistillTopK::from_examples(cfg, examples)), opts, control: Some(control) }
    }
}

impl ModelFit for DistillFit<'_, '_> {
    fn fit<M: Model>(&mut self, model: M) -> std::io::Result<(FitReport, M)> {
        let control = take_control(&mut self.control, "distillation")?;
        let distill = self.distill.take().ok_or_else(|| std::io::Error::other("a distillation fit trains one model"))?;
        fit_controlled(model, distill, self.opts, None, control)
    }
}

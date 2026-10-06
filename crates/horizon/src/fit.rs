// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Training a timeline model: the settings ([`TrainSpec`]), the run
//! ([`train`]) and what it reports ([`Report`]). The SDK's `TimelineModel::train`
//! and the `train` capability action are both this one function.
//!
//! Swedish Embedded AB implements long-horizon risk models trained and
//! checked on its clients' own records. If your team needs expertise in
//! training time-to-event models that stop on held-out evidence and can be
//! interrupted safely, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! Training early-stops on the held-out event NLL and keeps the best model. A
//! run can be cancelled between optimiser steps ([`Hooks::cancelled`]): it
//! returns [`TrainError::Cancelled`] and no model, so a partial run is never
//! mistaken for a finished one.

use std::cell::Cell;
use std::rc::Rc;

use crate::config::{Backbone, HorizonConfig, Mixer, StackConfig};
use crate::encode::{encode, Encoded};
use crate::saved::Saved;
use crate::timeline::Subject;
use crate::train::{event_nll, TimelineObjective};
use crate::vocab::{FitOptions, Vocab};
use crate::Horizon;

/// Subjects per batch when training and predicting, unless the spec says otherwise.
pub const DEFAULT_BATCH: u32 = 256;
/// Optimiser steps at most (early stopping usually ends sooner).
pub const DEFAULT_STEPS: u32 = 4000;
/// Peak learning rate.
pub const DEFAULT_LR: f32 = 1e-3;
/// Held-out evaluations without improvement before training stops.
pub const DEFAULT_PATIENCE: u32 = 8;
/// Steps between held-out evaluations.
pub const DEFAULT_EVAL_INTERVAL: u32 = 100;
/// Share of each subject's numeric values hidden for the value objective.
pub const DEFAULT_MASK_RATE: f64 = 0.3;

/// What to train: the outcome codes, the model's shape and the optimiser's
/// schedule. Only the codes are required; everything else has a default.
#[derive(Clone, Debug)]
pub struct TrainSpec {
    codes: Vec<String>,
    absorbing: Vec<String>,
    shape: Option<HorizonConfig>,
    additive: bool,
    knots: Option<Vec<f32>>,
    max_tokens: Option<u32>,
    batch: u32,
    steps: u32,
    lr: f32,
    patience: u32,
    eval_interval: u32,
    mask_rate: f64,
    seed: u64,
    vocab: FitOptions,
    forecasts: Option<(u32, f32)>,
    visits: Option<u32>,
    backbone: Option<Backbone>,
    next_events: Option<(Vec<String>, f32)>,
}

impl TrainSpec {
    /// Predict `codes`; `absorbing` (a subset) end follow-up for all of them.
    pub fn new<C: Into<String>, A: Into<String>>(
        codes: impl IntoIterator<Item = C>,
        absorbing: impl IntoIterator<Item = A>,
    ) -> TrainSpec {
        TrainSpec {
            codes: codes.into_iter().map(Into::into).collect(),
            absorbing: absorbing.into_iter().map(Into::into).collect(),
            shape: None,
            additive: false,
            knots: None,
            max_tokens: None,
            batch: DEFAULT_BATCH,
            steps: DEFAULT_STEPS,
            lr: DEFAULT_LR,
            patience: DEFAULT_PATIENCE,
            eval_interval: DEFAULT_EVAL_INTERVAL,
            mask_rate: DEFAULT_MASK_RATE,
            seed: 1,
            vocab: FitOptions::default(),
            forecasts: None,
            visits: None,
            backbone: None,
            next_events: None,
        }
    }
    /// Use exactly this model shape (its `vocab` and `n_codes` are replaced
    /// by the fitted vocabulary's and the spec's codes).
    pub fn shape(mut self, cfg: HorizonConfig) -> Self {
        self.shape = Some(cfg);
        self
    }
    /// Train the additive proportional-hazards baseline instead of the set encoder.
    pub fn additive(mut self, on: bool) -> Self {
        self.additive = on;
        self
    }
    /// Hazard piece boundaries over time since prediction, starting at 0.
    pub fn knots(mut self, knots: Vec<f32>) -> Self {
        self.knots = Some(knots);
        self
    }
    /// Tokens per subject, including the summary token.
    pub fn max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }
    /// Subjects per batch.
    pub fn batch(mut self, b: u32) -> Self {
        self.batch = b;
        self
    }
    /// Optimiser steps at most.
    pub fn steps(mut self, steps: u32) -> Self {
        self.steps = steps;
        self
    }
    /// Peak learning rate.
    pub fn lr(mut self, lr: f32) -> Self {
        self.lr = lr;
        self
    }
    /// Early-stopping patience, in held-out evaluations.
    pub fn patience(mut self, p: u32) -> Self {
        self.patience = p;
        self
    }
    /// Steps between held-out evaluations.
    pub fn eval_interval(mut self, n: u32) -> Self {
        self.eval_interval = n;
        self
    }
    /// Share of numeric values hidden for the value objective (0 turns it off).
    pub fn mask_rate(mut self, r: f64) -> Self {
        self.mask_rate = r;
        self
    }
    /// Seed of the initial weights, the batches and the masks.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
    /// The outcome codes being trained.
    pub fn codes(&self) -> &[String] {
        &self.codes
    }
    /// The seed set by [`TrainSpec::seed`].
    pub fn run_seed(&self) -> u64 {
        self.seed
    }
    /// Steps between held-out evaluations ([`TrainSpec::eval_interval`]).
    pub fn evaluation_interval(&self) -> u32 {
        self.eval_interval
    }
    /// Train a forecast head on up to `per_subject` future measurements of
    /// each subject (observations after its entry), weighted `weight`
    /// against the event objective; `TimelineModel::forecast` needs it.
    pub fn forecasts(mut self, per_subject: u32, weight: f32) -> Self {
        self.forecasts = Some((per_subject, weight));
        self
    }
    /// Carry a continuous-time state across the most recent `visits` visits
    /// (distinct observation times) instead of reading the whole history as
    /// one set: between visits the state reverts towards the population over
    /// the elapsed time, so gaps longer than any in training extrapolate.
    pub fn visits(mut self, visits: u32) -> Self {
        self.visits = Some(visits);
        self
    }
    /// What carries the visits to the prediction time with
    /// [`TrainSpec::visits`]: the continuous-time state (the default) or
    /// attention with rotary angles from real time.
    pub fn backbone(mut self, backbone: Backbone) -> Self {
        self.backbone = Some(backbone);
        self
    }
    /// Carry the visits (see [`TrainSpec::visits`], which this needs)
    /// through a stack of `blocks` residual blocks that mix the visit
    /// sequence as `mixer`: [`Mixer::Attention`], [`Mixer::GatedDeltaNet`]
    /// (a matrix-state delta rule whose decay is the exponential of the
    /// physical time between visits) or [`Mixer::Hybrid`] (three Gated
    /// DeltaNet blocks to every attention block). The same as
    /// [`TrainSpec::backbone`] with a [`Backbone::Stack`].
    pub fn mixer(self, mixer: Mixer, blocks: u32) -> Self {
        self.backbone(Backbone::Stack(StackConfig::new(mixer, blocks)))
    }
    /// Also model which of `events` (event codes, outcome codes or not) is the
    /// FIRST to happen after the prediction time, and when, weighted `weight`
    /// against the outcome codes: a self-supervised objective that teaches the
    /// state what comes next from every history, whether or not an outcome
    /// followed. `TimelineModel::predict_next_events` reads it back; the
    /// outcome codes keep their meaning and the held-out event NLL stays theirs.
    pub fn next_events<E: Into<String>>(
        mut self,
        events: impl IntoIterator<Item = E>,
        weight: f32,
    ) -> Self {
        self.next_events = Some((events.into_iter().map(Into::into).collect(), weight));
        self
    }
    /// Quantile knots per numeric variable, and the subjects a categorical
    /// level needs to get its own token.
    pub fn vocabulary(mut self, knots: usize, min_count: usize) -> Self {
        self.vocab = FitOptions { knots, min_count };
        self
    }

    /// The model configuration these settings give a model over `vocab`.
    pub fn config(&self, vocab: &Vocab) -> HorizonConfig {
        let mut cfg = self
            .shape
            .clone()
            .unwrap_or_else(|| HorizonConfig::default_for(vocab.len(), self.codes.len() as u32));
        cfg.vocab = vocab.len();
        cfg.n_codes = vocab.head_codes() as u32;
        cfg.next_codes = vocab.next_events.len() as u32;
        if let Some((_, w)) = &self.next_events {
            cfg.next_weight = *w;
        }
        cfg.additive = self.additive || cfg.additive;
        if let Some(k) = &self.knots {
            cfg.knots = k.clone();
        }
        if let Some(n) = self.max_tokens {
            cfg.max_tokens = n;
        }
        if let Some((n, w)) = self.forecasts {
            cfg.forecasts = n;
            cfg.forecast_weight = w;
        }
        if let Some(v) = self.visits {
            cfg.visits = v;
        }
        if let Some(b) = self.backbone {
            cfg.backbone = b;
        }
        cfg
    }
}

/// What training did.
#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    /// Optimiser steps run (early stopping may end before the spec's limit).
    pub steps: u32,
    /// Training loss before the first step.
    pub initial_loss: f32,
    /// Training loss at the last step.
    pub final_loss: Option<f32>,
    /// Weighted event NLL on the held-out subjects, of the model kept.
    pub held_out_event_nll: f32,
    /// Trainable parameters.
    pub parameters: usize,
    /// Known tokens left out because a subject had more than `max_tokens - 1`.
    pub truncated_tokens: usize,
}

/// A progress report, one per evaluation interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepProgress {
    /// Optimiser steps completed.
    pub step: u32,
    /// The run's step limit.
    pub steps: u32,
    /// The last step's training loss.
    pub loss: f32,
    /// The most recent held-out event NLL, once one was taken.
    pub held_out_event_nll: Option<f32>,
}

/// What the caller of [`train`] sees and controls while it runs.
#[derive(Default)]
pub struct Hooks<'a> {
    /// Called every [`TrainSpec::eval_interval`] steps and at the step limit.
    pub progress: Option<&'a mut dyn FnMut(&StepProgress)>,
    /// Polled after every optimiser step; `true` stops the run.
    pub cancelled: Option<&'a dyn Fn() -> bool>,
}

/// Why training produced no model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TrainError {
    /// [`Hooks::cancelled`] fired: the run was stopped and its partial work
    /// discarded.
    Cancelled,
    /// The data, the settings or the device refused.
    Failed(String),
}

impl std::fmt::Display for TrainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrainError::Cancelled => f.write_str("cancelled"),
            TrainError::Failed(why) => f.write_str(why),
        }
    }
}

impl std::error::Error for TrainError {}

impl From<String> for TrainError {
    fn from(why: String) -> Self {
        TrainError::Failed(why)
    }
}

/// Fit the vocabulary of `spec` on `train`: what [`train`] does first, kept
/// apart so an ensemble can fit it once for all its members.
pub fn fit_vocab(train: &[Subject], spec: &TrainSpec) -> Result<Vocab, String> {
    let mut vocab = Vocab::fit(train, &spec.codes, &spec.absorbing, &spec.vocab)?;
    if let Some((events, _)) = &spec.next_events {
        vocab = vocab.with_next_events(events)?;
    }
    Ok(vocab)
}

/// Train on `train` under `vocab` (from [`fit_vocab`]), early-stopping on the
/// event NLL of `held_out` (which must not overlap `train`) and keeping the
/// best model, with the support of `train` recorded beside it.
pub fn train_with_vocab(
    train: &[Subject],
    held_out: &[Subject],
    spec: &TrainSpec,
    vocab: Vocab,
    hooks: &mut Hooks<'_>,
) -> Result<(Saved, Report), TrainError> {
    if train.is_empty() || held_out.is_empty() {
        return Err("training and held-out subjects are both required".to_string().into());
    }
    // The training subjects fixed the units; held-out ones must agree.
    for s in held_out {
        vocab.check_units(s)?;
    }
    let cfg = spec.config(&vocab);
    cfg.validate()?;
    let enc_train: Vec<Encoded> = train.iter().map(|s| encode(s, &vocab, &cfg)).collect();
    let enc_held: Vec<Encoded> = held_out.iter().map(|s| encode(s, &vocab, &cfg)).collect();
    let truncated_tokens = enc_train.iter().chain(&enc_held).map(|e| e.truncated).sum();
    let model = Horizon::new(cfg.clone(), spec.batch, &crate::init_weights(&cfg, spec.seed));
    let opts = model::FitOpts {
        steps: spec.steps,
        batch_size: spec.batch,
        block_size: cfg.max_tokens,
        lr: spec.lr,
        min_lr: spec.lr / 10.0,
        warmup: (spec.steps / 40).max(1),
        decay_iters: spec.steps,
        weight_decay: 0.1,
        eval_interval: spec.eval_interval,
        patience: spec.patience,
        checkpoint_secs: 0,
        seed: spec.seed,
        ..Default::default()
    };
    let last_held_out = Rc::new(Cell::new(None));
    let objective = TimelineObjective::new(&enc_train, Some(&enc_held), spec.mask_rate)
        .report_held_out_to(last_held_out.clone());
    let every = spec.eval_interval.max(1);
    let mut cancelled = false;
    let mut on_step = |s: &model::StepReport| {
        if hooks.cancelled.is_some_and(|c| c()) {
            cancelled = true;
            return false;
        }
        if s.step.is_multiple_of(every) || s.step == s.steps {
            if let Some(progress) = hooks.progress.as_mut() {
                progress(&StepProgress {
                    step: s.step,
                    steps: s.steps,
                    loss: s.loss,
                    held_out_event_nll: last_held_out.get(),
                });
            }
        }
        true
    };
    let control = model::FitControl { on_step: Some(&mut on_step), ..Default::default() };
    let (fit, model) = model::fit_controlled(model, objective, &opts, None, control)
        .map_err(|e| TrainError::Failed(e.to_string()))?;
    if cancelled {
        return Err(TrainError::Cancelled);
    }
    let held_out_event_nll = event_nll(&model, &enc_held);
    let report = Report {
        steps: fit.steps_completed,
        initial_loss: fit.initial_loss,
        final_loss: fit.final_loss,
        held_out_event_nll,
        parameters: cfg.param_list().iter().map(|(_, n)| n).sum(),
        truncated_tokens,
    };
    let mut saved = Saved::new(model, vocab);
    saved.fit_support(train)?;
    Ok((saved, report))
}

/// [`train_with_vocab`] with the vocabulary fitted on `train`.
pub fn train(
    train: &[Subject],
    held_out: &[Subject],
    spec: &TrainSpec,
    hooks: &mut Hooks<'_>,
) -> Result<(Saved, Report), TrainError> {
    if train.is_empty() || held_out.is_empty() {
        return Err("training and held-out subjects are both required".to_string().into());
    }
    let vocab = fit_vocab(train, spec)?;
    train_with_vocab(train, held_out, spec, vocab, hooks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_spec_applies_only_what_it_was_given() {
        let s = TrainSpec::new(["a", "b"], ["a"])
            .knots(vec![0.0, 1.0, 2.0])
            .steps(10);
        let subjects = crate::synthetic::population(50, 1).0;
        let vocab = Vocab::fit(&subjects, &s.codes, &s.absorbing, &s.vocab).unwrap();
        let cfg = s.config(&vocab);
        assert_eq!(cfg.knots, vec![0.0, 1.0, 2.0]);
        assert_eq!(cfg.n_codes, 2);
        assert_eq!(
            cfg.d_model,
            HorizonConfig::default_for(1, 1).d_model,
            "unset fields keep the default"
        );
        assert!(!cfg.additive);
        assert!(s.clone().additive(true).config(&vocab).additive);
    }
}

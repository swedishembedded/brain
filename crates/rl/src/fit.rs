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
//! * [`PolicyGradientFit`] - the same clipped objective on rollouts collected outside brain.
//! * [`TeacherFit`] - the top-K distributions a teacher gives a prepared dataset, for [`DistillFit`].
//! * [`DistillFit`] - top-K forward-KL distillation from teacher distributions.
//!   Every pass is a weighted cross-entropy, which `Model` computes through the
//!   chunked LM head (`model::head_ce`), so the logits of a whole vocabulary
//!   are never held for a whole row.

use model::finetune::ModelFit;
use model::{fit_controlled, FitControl, FitOpts, FitReport, Model};

use crate::env::{Environment, Verifier};
use crate::objective::distill::{DistillConfig, DistillExample, DistillTopK, TopK};
use crate::objective::dpo::{Dpo, DpoConfig, PackedPair};
use crate::objective::external::{ExternalPolicyGradient, PolicyGradientConfig, PolicyGradientStats};
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

/// The distillation examples of one prepared-dataset row (`tokens`, with
/// `targets[t]` the token after `tokens[t]` or IGNORE): the prompt runs
/// through the first supervised position, the completion to the last, and
/// `logits` (`[positions, vocab]`, from the row's first position) gives each
/// supervised position its teacher's top `k`. An unsupervised position inside
/// the completion (the user's turn between two answers) carries no teacher
/// signal. `None` for a row with nothing supervised.
pub fn teacher_example(tokens: &[u32], targets: &[u32], logits: &[f32], vocab: usize, k: usize) -> Option<DistillExample> {
    let supervised = |t: usize| targets[t] != model::IGNORE;
    let first = (0..targets.len()).find(|&t| supervised(t))?;
    let last = (0..targets.len()).rev().find(|&t| supervised(t))?;
    let completion: Vec<u32> = (first..=last).map(|t| if supervised(t) { targets[t] } else { tokens[t + 1] }).collect();
    let teacher = (first..=last).map(|t| if supervised(t) { TopK::from_logits(&logits[t * vocab..(t + 1) * vocab], k) } else { TopK::default() }).collect();
    Some(DistillExample { prompt: tokens[..=first].to_vec(), completion, teacher })
}

/// The teacher's top-`k` distributions over the train split of the prepared
/// dataset at `dir` ([`teacher_example`] per record), from whichever model the
/// trainer builds: the examples a [`DistillFit`] then trains a student on.
/// The model must be built for one row (`opts.batch_size` 1): the logits of a
/// whole record come from [`Model::logits_all`].
pub struct TeacherFit<'a> {
    pub dir: &'a std::path::Path,
    pub opts: &'a FitOpts,
    pub k: usize,
    /// The examples, once the visit has run.
    pub examples: Vec<DistillExample>,
}

impl ModelFit for TeacherFit<'_> {
    fn fit<M: Model>(&mut self, model: M) -> std::io::Result<(FitReport, M)> {
        let (train, _, cfg, vocab) = model::load_dataset(self.dir, self.opts)?;
        let one = data::loader::BatchConfig { batch_size: 1, ..cfg };
        for (x, targets) in model::example_rows(&train, &one) {
            let Some(last) = targets.iter().rposition(|&t| t != model::IGNORE) else { continue };
            let logits = model.logits_all(&x[..=last]).ok_or_else(|| std::io::Error::other("the teacher has no logits"))?;
            self.examples.extend(teacher_example(&x, &targets, &logits, vocab as usize, self.k));
        }
        if self.examples.is_empty() {
            return Err(std::io::Error::other("the prepared dataset has no supervised record to distill"));
        }
        let report = FitReport { initial_loss: 0.0, final_loss: None, steps_completed: 0, resumed_at: None, interrupted: false, evaluations: Vec::new(), kept_best: None, stopped_early: false };
        Ok((report, model))
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

/// Policy-gradient training on externally collected rollouts over whichever model the trainer builds
/// ([`ExternalPolicyGradient`]): the objective is opened on the prepared dataset when the model arrives, and what the
/// run measured is kept for [`Self::stats`].
pub struct PolicyGradientFit<'a, 'c> {
    dir: &'a std::path::Path,
    config: PolicyGradientConfig,
    opts: &'a FitOpts,
    control: Option<FitControl<'c>>,
    stats: Option<PolicyGradientStats>,
}

impl<'a, 'c> PolicyGradientFit<'a, 'c> {
    /// `dir` is the prepared dataset ([`data::chat::prepare_chat_samples`]) of the rollouts.
    pub fn new(dir: &'a std::path::Path, config: PolicyGradientConfig, opts: &'a FitOpts, control: FitControl<'c>) -> PolicyGradientFit<'a, 'c> {
        PolicyGradientFit { dir, config, opts, control: Some(control), stats: None }
    }

    /// What the run measured, after it ran (interrupted or not); `None` before.
    pub fn stats(&self) -> Option<&PolicyGradientStats> {
        self.stats.as_ref()
    }

    /// [`Self::stats`], consuming the visitor.
    pub fn into_stats(self) -> Option<PolicyGradientStats> {
        self.stats
    }
}

impl ModelFit for PolicyGradientFit<'_, '_> {
    fn fit<M: Model>(&mut self, model: M) -> std::io::Result<(FitReport, M)> {
        let control = take_control(&mut self.control, "policy-gradient")?;
        let mut objective = ExternalPolicyGradient::open(self.dir, self.opts, self.config)?;
        let result = fit_controlled(model, &mut objective, self.opts, None, control);
        self.stats = Some(objective.into_stats());
        result
    }
}

#[cfg(test)]
mod teacher_tests {
    use super::*;

    /// tokens 10..16; positions 2 and 4 are supervised, 3 (between them) is not.
    #[test]
    fn a_teacher_example_spans_the_supervised_positions() {
        let tokens = [10, 11, 12, 13, 14, 15];
        let i = model::IGNORE;
        let targets = [i, i, 13, i, 15, i];
        let vocab = 20;
        let mut logits = vec![0f32; 5 * vocab];
        logits[2 * vocab + 13] = 9.0;
        logits[4 * vocab + 15] = 9.0;
        let ex = teacher_example(&tokens, &targets, &logits, vocab, 2).expect("a supervised row");
        assert_eq!(ex.prompt, [10, 11, 12], "the prompt runs through the first supervised position");
        assert_eq!(ex.completion, [13, 14, 15], "the completion runs to the last, the gap's own token kept");
        assert_eq!(ex.teacher[0].ids[0], 13);
        assert!(ex.teacher[1].ids.is_empty(), "an unsupervised position carries no teacher signal");
        assert_eq!(ex.teacher[2].ids[0], 15);
        assert!((ex.teacher[0].probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(teacher_example(&tokens, &[i; 6], &logits, vocab, 2).is_none());
    }
}

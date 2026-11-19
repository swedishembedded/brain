// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! KTO: fine-tuning on unpaired feedback. Each example is one prompt and one completion marked desirable or
//! undesirable, and no second completion is needed to compare it with (Ethayarajh et al., "KTO: Model Alignment as
//! Prospect Theoretic Optimization", 2024).
//!
//! Swedish Embedded AB implements solutions for aligning a model with a product's own thumbs-up and thumbs-down
//! feedback, for its clients. If your team needs expertise in preference and feedback fine-tuning of language models
//! then you can procure our services by sending an email to info@swedishembedded.com.
//!
//! ## The objective
//!
//! With `r = log pi(y | x) - log ref(y | x)` the example's log-ratio (each log-probability summed over the
//! completion's supervised tokens) and `z` the reference point, the batch's estimate of `KL(pi || ref)`:
//!
//! * a desirable example costs `lambda_D * (1 - sigma(beta * (r - z)))`,
//! * an undesirable one `lambda_U * (1 - sigma(beta * (z - r)))`.
//!
//! `z` is `max(0, mean of r')` over the step's mismatched pairs, `r'` the log-ratio of an example's prompt followed
//! by another example's completion, and is held fixed (no gradient flows through it), as the paper takes it. Each of
//! an example's supervised tokens then has one weight: since `r` is a sum of log-probabilities, `dL/dCE_t = -dL/dr`,
//! the same for every token of the row, as in DPO ([`super::dpo`]). A desirable example's weight is
//! `lambda_D * beta * sigma(u) * sigma(-u)` with `u = beta * (r - z)` (it raises the completion), an undesirable
//! one's `-lambda_U * beta * sigma(u) * sigma(-u)` with `u = beta * (z - r)` (it lowers it), times the forward's
//! active-position count the model's weighted cross-entropy divides by.
//!
//! ## Two rows, each its own pass
//!
//! An example is packed as two rows of `seq_len`: its own completion (trained) and its prompt with the next example's
//! completion (forwarded for `z` only). The model is built at `b = 1`: a step of `k` micro-batches forwards its `k`
//! mismatched rows to measure `z`, then forwards and backwards each own row, so no pass computes a row that trains
//! nothing. The own row's loss and gradient are divided by both rows' supervised count (the count the two-row batch
//! this replaces divided by), so the term and its gradient are unchanged. The reference is the model the run starts
//! from, so no second model is held. An example is scored when a step first draws it, under the frozen start: once
//! the weights have moved, a copy of the start-of-run adapter tensors is put in while it is scored
//! ([`super::pg_batch::AdapterState`], as the policy-gradient objective does). A run never pays for examples it does not
//! draw, and its first step starts at once. A model without adapters has no start to put back, so it scores every
//! example before the first step ([`Objective::check_data`]).

use data::rng::Rng;
use model::training::Unsupported;
use model::{Batch, Model, ModelConfig, Objective, StepVerdict, IGNORE};

use super::dpo::row_sum;
use super::pg_batch::AdapterState;

/// The default `beta` (the paper's).
pub const DEFAULT_KTO_BETA: f32 = 0.1;
/// The default weights of the desirable and of the undesirable examples' terms.
pub const DEFAULT_KTO_DESIRABLE_WEIGHT: f32 = 1.0;
pub const DEFAULT_KTO_UNDESIRABLE_WEIGHT: f32 = 1.0;

/// The rows a KTO model is built with: each row of an example is its own pass.
pub const FEEDBACK_ROWS: u32 = 1;

/// KTO's hyperparameters. `seq_len` is one row's length; the model is built at `b = 1`, `t = seq_len`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeedbackConfig {
    /// Scales the log-ratio's distance from the reference point inside the sigmoid.
    pub beta: f32,
    /// `lambda_D`: the weight of a desirable example's term.
    pub desirable_weight: f32,
    /// `lambda_U`: the weight of an undesirable example's term.
    pub undesirable_weight: f32,
    pub seq_len: usize,
}

impl FeedbackConfig {
    /// Refuse what no run can train: a `beta` that is not finite and positive, a weight that is negative or not
    /// finite, both weights zero, a row with no position that predicts a token.
    pub fn validate(&self) -> Result<(), String> {
        if !(self.beta.is_finite() && self.beta > 0.0) {
            return Err(format!("beta must be finite and positive, got {}", self.beta));
        }
        for (name, w) in [("desirable_weight", self.desirable_weight), ("undesirable_weight", self.undesirable_weight)] {
            if !(w.is_finite() && w >= 0.0) {
                return Err(format!("{name} must be finite and not negative, got {w}"));
            }
        }
        if self.desirable_weight == 0.0 && self.undesirable_weight == 0.0 {
            return Err("desirable_weight and undesirable_weight are both zero, so no example would train".to_string());
        }
        if self.seq_len < 2 {
            return Err(format!("seq_len {} leaves no position that predicts a token", self.seq_len));
        }
        Ok(())
    }
}

impl model::training_plan::PlanCheck for FeedbackConfig {
    fn plan_name(&self) -> &'static str {
        "feedback"
    }

    fn check(&self, opts: &model::FitOpts) -> Result<(), String> {
        self.validate()?;
        if self.seq_len != opts.block_size as usize {
            return Err(format!("rows are packed at seq_len {} but the model is built for the block size {}", self.seq_len, opts.block_size));
        }
        Ok(())
    }
}

/// One example's term: the weight of each supervised token of its own row and the loss it is the gradient of.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeedbackTerm {
    pub weight: f32,
    pub loss: f32,
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// The term of an example with log-ratio `logratio` against the reference point `z` (see the module doc), for a
/// loss divided by `count` active positions (both rows'). Evaluated in f64.
pub fn feedback_term(cfg: &FeedbackConfig, desirable: bool, logratio: f32, z: f32, count: f32) -> FeedbackTerm {
    let beta = f64::from(cfg.beta);
    let (lambda, sign) = if desirable { (f64::from(cfg.desirable_weight), 1.0) } else { (f64::from(cfg.undesirable_weight), -1.0) };
    let u = sign * beta * (f64::from(logratio) - f64::from(z));
    let value = lambda * sigmoid(u);
    // d(lambda - value)/dr = -sign * lambda * beta * s(1 - s); the token weight is -dL/dr times the count.
    let slope = lambda * beta * sigmoid(u) * sigmoid(-u);
    FeedbackTerm { weight: (sign * slope * f64::from(count)) as f32, loss: (lambda - value) as f32 }
}

/// The reference point a step's mismatched log-ratios give: their mean, never below zero.
pub fn reference_point(mismatched: &[f32]) -> f32 {
    if mismatched.is_empty() {
        return 0.0;
    }
    let mean = mismatched.iter().map(|&r| f64::from(r)).sum::<f64>() / mismatched.len() as f64;
    mean.max(0.0) as f32
}

/// One row of an example: `seq_len` tokens, the targets of its supervised positions, and the frozen reference's
/// log-probability of each target.
#[derive(Clone, Debug, PartialEq)]
struct FeedbackRow {
    tokens: Vec<u32>,
    targets: Vec<u32>,
    ref_lp: Vec<f32>,
    /// Supervised positions.
    count: usize,
    /// `ref_lp` holds the reference's log-probabilities.
    scored: bool,
}

impl FeedbackRow {
    fn pack(seq_len: usize, name: &str, (ids, mask): (&[u32], &[bool])) -> Result<FeedbackRow, String> {
        if ids.len() != mask.len() {
            return Err(format!("{name}: {} token ids but {} mask entries", ids.len(), mask.len()));
        }
        if ids.len() > seq_len {
            return Err(format!("{name}: {} tokens do not fit a row of {seq_len}", ids.len()));
        }
        let mut tokens = vec![0u32; seq_len];
        let mut targets = vec![IGNORE; seq_len];
        tokens[..ids.len()].copy_from_slice(ids);
        for t in 1..ids.len() {
            if mask[t] {
                targets[t - 1] = ids[t];
            }
        }
        let count = targets.iter().filter(|&&y| y != IGNORE).count();
        if count == 0 {
            return Err(format!("{name}: no supervised token after the first, so the row has no log-probability"));
        }
        Ok(FeedbackRow { tokens, targets, ref_lp: vec![0f32; seq_len], count, scored: false })
    }

    /// Forward the row alone and return the log-probability of each position's target.
    fn forward<M: Model>(&self, model: &M) -> Vec<f32> {
        model.set_batch(Batch::Lm { tokens: &self.tokens, targets: &self.targets });
        let _ = model.forward();
        model::training::objectives(model).batch_token_logprobs()
    }

    /// The row's log-ratio from the log-probabilities of a forward of it.
    fn logratio(&self, new_lp: &[f32]) -> f32 {
        let (new, reference, _) = row_sum(new_lp, &self.ref_lp, &self.targets, 0, self.tokens.len());
        new - reference
    }

    fn score_reference<M: Model>(&mut self, reference: &M) {
        let lp = self.forward(reference);
        for (i, &y) in self.targets.iter().enumerate() {
            self.ref_lp[i] = if y == IGNORE { 0.0 } else { lp[i] };
        }
        self.scored = true;
    }
}

/// One example: its own completion and its prompt with another example's completion, each a row of `seq_len`, with the
/// frozen reference's log-probability of every supervised target.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedFeedback {
    own: FeedbackRow,
    mismatched: FeedbackRow,
    desirable: bool,
}

impl PackedFeedback {
    /// Pack two rendered conversations (token ids and a supervision mask each): `own` the example, `mismatched` its
    /// prompt followed by another example's completion. Position `t` predicts token `t + 1` and is a target exactly
    /// when that token is supervised. Refused, never truncated: a conversation longer than `seq_len`, a mask of another
    /// length than its ids, a row with no supervised token after the first.
    pub fn from_masked(seq_len: usize, own: (&[u32], &[bool]), mismatched: (&[u32], &[bool]), desirable: bool) -> Result<PackedFeedback, String> {
        Ok(PackedFeedback { own: FeedbackRow::pack(seq_len, "completion", own)?, mismatched: FeedbackRow::pack(seq_len, "mismatched completion", mismatched)?, desirable })
    }

    /// Make `reference` the frozen reference of both rows: one forward of each, keeping its log-probability of every
    /// target.
    pub fn score_reference<M: Model>(&mut self, reference: &M) {
        self.own.score_reference(reference);
        self.mismatched.score_reference(reference);
    }

    pub fn desirable(&self) -> bool {
        self.desirable
    }

    /// Whether the reference has been scored ([`Self::score_reference`]).
    pub fn is_scored(&self) -> bool {
        self.own.scored && self.mismatched.scored
    }

    /// The reference's log-probability of each supervised target of the example's own row, in position order; `None`
    /// before the reference is scored.
    pub fn reference_logprobs(&self) -> Option<Vec<f32>> {
        self.is_scored().then(|| self.own.targets.iter().zip(&self.own.ref_lp).filter(|(&y, _)| y != IGNORE).map(|(_, &lp)| lp).collect())
    }

    fn seq_len(&self) -> usize {
        self.own.tokens.len()
    }

    /// Refuse a token id the model has no embedding row for.
    pub fn check_vocab(&self, vocab: u32) -> Result<(), String> {
        let ids = |row: &FeedbackRow| row.tokens.iter().copied().chain(row.targets.iter().copied().filter(|&y| y != IGNORE)).collect::<Vec<u32>>();
        match ids(&self.own).into_iter().chain(ids(&self.mismatched)).find(|&id| id >= vocab) {
            Some(id) => Err(format!("token id {id} is outside the model's vocabulary of {vocab}")),
            None => Ok(()),
        }
    }
}

/// The log-ratios of both rows of `packed` under `model` (each forwarded here): its own and the mismatched one.
pub fn logratios<M: Model>(model: &M, packed: &PackedFeedback) -> (f32, f32) {
    (packed.own.logratio(&packed.own.forward(model)), mismatched_logratio(model, packed))
}

/// The log-ratio of the mismatched row of `packed` under `model` (forwarded here): one term of the reference point.
pub fn mismatched_logratio<M: Model>(model: &M, packed: &PackedFeedback) -> f32 {
    packed.mismatched.logratio(&packed.mismatched.forward(model))
}

/// One example's training forward against the held reference point `z`: its own row forwarded, the term, and its
/// weight installed on the row's supervised positions, with the loss divided by both rows' supervised count -
/// everything a micro-step does before its backward. Also returns the example's log-ratio. `model` is built at
/// `b = 1`, `t = seq_len`, with weighted loss enabled.
pub fn weigh_feedback<M: Model>(cfg: &FeedbackConfig, model: &M, packed: &PackedFeedback, z: f32) -> (FeedbackTerm, f32) {
    let row = &packed.own;
    let new_lp = row.forward(model);
    let objectives = model::training::objectives(model);
    let count = (row.count + packed.mismatched.count) as f32;
    objectives.set_loss_denominator(count);
    let logratio = row.logratio(&new_lp);
    let term = feedback_term(cfg, packed.desirable, logratio, z, count);
    let weights: Vec<f32> = row.targets.iter().map(|&y| if y == IGNORE { 0.0 } else { term.weight }).collect();
    objectives.set_loss_weights(&weights);
    (term, logratio)
}

/// Why a [`Feedback`] objective cannot be built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FeedbackConfigError {
    /// [`FeedbackConfig::validate`] refused the configuration.
    Invalid(String),
    /// The dataset holds fewer than two examples: the reference point is estimated from mismatched pairs.
    TooFewExamples(usize),
    /// Example `index` is packed at `got` positions, not the configuration's `expected`.
    Width { index: usize, got: usize, expected: usize },
}

impl std::fmt::Display for FeedbackConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FeedbackConfigError::Invalid(why) => write!(f, "feedback: {why}"),
            FeedbackConfigError::TooFewExamples(n) => write!(f, "feedback: {n} example(s); the reference point is estimated from mismatched pairs, which needs at least two"),
            FeedbackConfigError::Width { index, got, expected } => write!(f, "feedback: example {index} is packed at seq_len {got}, the configuration says {expected}"),
        }
    }
}

impl std::error::Error for FeedbackConfigError {}

impl From<FeedbackConfigError> for std::io::Error {
    fn from(error: FeedbackConfigError) -> Self {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, error)
    }
}

/// The KTO objective over a fixed feedback dataset, referenced against the model the run starts from. Generic over any
/// model with token objectives.
pub struct Feedback {
    cfg: FeedbackConfig,
    examples: Vec<PackedFeedback>,
    /// The start-of-run adapter tensors ([`Objective::prepare`]); `None` for a model without adapters, which is scored
    /// whole before the first step.
    start: Option<AdapterState>,
    /// An update was applied since the start (also true of a resumed run), so the live adapter is not the start's.
    weights_moved: bool,
    vocab: Option<u32>,
    trained: u32,
    last_z: f32,
    last_logratio: f32,
}

impl Feedback {
    /// Every example must be packed at `cfg.seq_len`, and there must be at least two.
    pub fn try_from_dataset(cfg: FeedbackConfig, examples: Vec<PackedFeedback>) -> Result<Feedback, FeedbackConfigError> {
        cfg.validate().map_err(FeedbackConfigError::Invalid)?;
        if examples.len() < 2 {
            return Err(FeedbackConfigError::TooFewExamples(examples.len()));
        }
        if let Some((index, e)) = examples.iter().enumerate().find(|(_, e)| e.seq_len() != cfg.seq_len) {
            return Err(FeedbackConfigError::Width { index, got: e.seq_len(), expected: cfg.seq_len });
        }
        Ok(Feedback { cfg, examples, start: None, weights_moved: false, vocab: None, trained: 0, last_z: 0.0, last_logratio: 0.0 })
    }
}

impl Feedback {
    /// Each example's log-ratio of its own completion under `model` against the reference it was scored with; `None`
    /// for an example no step drew, whose reference was never scored.
    pub fn logratios<M: Model>(&self, model: &M) -> Vec<Option<f32>> {
        self.examples.iter().map(|e| e.is_scored().then(|| e.own.logratio(&e.own.forward(model)))).collect()
    }

    /// Every example's [`PackedFeedback::reference_logprobs`], in example order.
    pub fn reference_logprobs(&self) -> Vec<Option<Vec<f32>>> {
        self.examples.iter().map(PackedFeedback::reference_logprobs).collect()
    }

    /// Score the reference of every example in `drawn` that has none yet, under the start-of-run policy.
    fn ensure_scored<M: Model>(&mut self, model: &M, drawn: &[usize]) {
        let mut todo: Vec<usize> = drawn.iter().copied().filter(|&e| !self.examples[e].is_scored()).collect();
        todo.sort_unstable();
        todo.dedup();
        if todo.is_empty() {
            return;
        }
        let examples = &mut self.examples;
        let mut score = || {
            for &e in &todo {
                examples[e].score_reference(model);
            }
        };
        match (&self.start, self.weights_moved) {
            (Some(start), true) => start.with_applied(model, score),
            (_, false) => score(),
            (None, true) => unreachable!("a model without adapters is scored whole in check_data"),
        }
    }
}

impl<M: Model> Objective<M> for Feedback {
    fn regime(&self) -> &'static str {
        "feedback"
    }

    /// Opt into the weighted loss and keep the start-of-run adapter tensors the reference is scored under. Runs before
    /// a resumed run restores its weights, so the start captured is the run's own.
    fn prepare(&mut self, model: &mut M) -> Result<(), Unsupported> {
        model::training::require_token_objectives(model)?.enable_weighted_loss();
        let start = AdapterState::capture(model);
        self.start = (!start.is_empty()).then_some(start);
        Ok(())
    }

    /// The weights differ from the start-of-run ones exactly when an update was applied (also true of a resumed run).
    fn updates_applied(&mut self, applied: u32) {
        self.weights_moved = applied > 0;
    }

    fn check_data(&mut self, model: &M) -> Result<(), std::io::Error> {
        let vocab = model.config().vocab();
        self.vocab = Some(vocab);
        for (index, e) in self.examples.iter().enumerate() {
            e.check_vocab(vocab).map_err(|why| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("feedback example {index}: {why}")))?;
        }
        if self.start.is_none() {
            let all: Vec<usize> = (0..self.examples.len()).collect();
            self.ensure_scored(model, &all);
        }
        Ok(())
    }

    fn micro_step(&mut self, model: &M, rng: &mut Rng) -> f32 {
        Objective::<M>::micro_steps(self, model, rng, 1)
    }

    /// Draw the step's examples, measure the reference point on their mismatched rows, then train each against it.
    fn micro_steps(&mut self, model: &M, rng: &mut Rng, k: u32) -> f32 {
        let drawn: Vec<usize> = (0..k.max(1)).map(|_| rng.gen_range_inclusive(0, self.examples.len() as i64 - 1) as usize).collect();
        self.ensure_scored(model, &drawn);
        let mismatched: Vec<f32> = drawn.iter().map(|&e| mismatched_logratio(model, &self.examples[e])).collect();
        let z = reference_point(&mismatched);
        let mut loss = 0.0;
        for &e in &drawn {
            let (term, logratio) = weigh_feedback(&self.cfg, model, &self.examples[e], z);
            model.backward();
            loss += term.loss;
            self.last_logratio = logratio;
            self.trained += 1;
        }
        self.last_z = z;
        loss
    }

    /// The loss of one example drawn from the probe's generator against its own mismatched row's reference point.
    fn loss_probe(&mut self, model: &M, rng: &mut Rng) -> f32 {
        let e = rng.gen_range_inclusive(0, self.examples.len() as i64 - 1) as usize;
        self.ensure_scored(model, &[e]);
        let z = reference_point(&[mismatched_logratio(model, &self.examples[e])]);
        weigh_feedback(&self.cfg, model, &self.examples[e], z).0.loss
    }

    fn step_verdict(&mut self) -> StepVerdict {
        match std::mem::take(&mut self.trained) {
            0 => StepVerdict::Skip("no feedback example this step".into()),
            _ => StepVerdict::Apply,
        }
    }

    fn metrics(&self) -> Vec<(&'static str, f32)> {
        vec![("kto_reference_point", self.last_z), ("kto_logratio", self.last_logratio)]
    }

    /// Each row is its own pass.
    fn batch_shape(&self) -> Option<(u32, u32)> {
        Some((FEEDBACK_ROWS, self.cfg.seq_len as u32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CFG: FeedbackConfig = FeedbackConfig { beta: 0.5, desirable_weight: 1.0, undesirable_weight: 1.5, seq_len: 8 };

    /// The fp64 oracle: the paper's loss, written out.
    fn oracle(desirable: bool, r: f64, z: f64) -> f64 {
        let s = |x: f64| 1.0 / (1.0 + (-x).exp());
        let beta = f64::from(CFG.beta);
        if desirable {
            f64::from(CFG.desirable_weight) * (1.0 - s(beta * (r - z)))
        } else {
            f64::from(CFG.undesirable_weight) * (1.0 - s(beta * (z - r)))
        }
    }

    /// L1: the loss is the paper's, and the weight is `count` times minus its derivative in the log-ratio, which is
    /// the derivative in each token's cross-entropy (checked by central differences of the oracle), at log-ratios far
    /// below, at and far above the reference point; a desirable example's weight raises the completion, an
    /// undesirable one's lowers it.
    #[test]
    fn the_term_is_the_papers_loss_and_its_weight_is_its_gradient() {
        let count = 7.0f32;
        for desirable in [true, false] {
            for (r, z) in [(-40.0f32, 0.3f32), (-1.2, 0.0), (0.3, 0.3), (0.9, 0.2), (2.5, 0.0), (60.0, 1.0)] {
                let t = feedback_term(&CFG, desirable, r, z, count);
                let want = oracle(desirable, f64::from(r), f64::from(z));
                assert!((f64::from(t.loss) - want).abs() < 1e-6, "{desirable} r {r} z {z}: {} vs {want}", t.loss);
                let h = 1e-5;
                let dl_dr = (oracle(desirable, f64::from(r) + h, f64::from(z)) - oracle(desirable, f64::from(r) - h, f64::from(z))) / (2.0 * h);
                // r = sum of log-probabilities = -sum of cross-entropies: dL/dCE_t = -dL/dr, scaled by the count.
                let want_weight = -dl_dr * f64::from(count);
                assert!((f64::from(t.weight) - want_weight).abs() <= 1e-6 * want_weight.abs().max(1e-3), "{desirable} r {r} z {z}: weight {} vs {want_weight}", t.weight);
                assert!(if desirable { t.weight >= 0.0 } else { t.weight <= 0.0 }, "{desirable}: {t:?}");
            }
        }
        // At the reference point the term is half its weight: lambda * (1 - sigma(0)).
        assert!((feedback_term(&CFG, false, 0.4, 0.4, 1.0).loss - 0.75).abs() < 1e-6);
    }

    /// The reference point is the mean of the mismatched log-ratios, clamped at zero (a KL is not negative).
    #[test]
    fn the_reference_point_is_the_clamped_mean_of_the_mismatched_log_ratios() {
        assert_eq!(reference_point(&[0.2, 0.6, -0.2]), 0.2);
        assert_eq!(reference_point(&[-0.5, -0.1]), 0.0);
        assert_eq!(reference_point(&[]), 0.0);
    }

    #[test]
    fn a_configuration_or_dataset_that_cannot_train_is_refused_by_name() {
        assert!(CFG.validate().is_ok());
        for (field, cfg) in [
            ("beta", FeedbackConfig { beta: 0.0, ..CFG }),
            ("beta", FeedbackConfig { beta: f32::NAN, ..CFG }),
            ("desirable_weight", FeedbackConfig { desirable_weight: -1.0, ..CFG }),
            ("undesirable_weight", FeedbackConfig { undesirable_weight: f32::INFINITY, ..CFG }),
            ("both zero", FeedbackConfig { desirable_weight: 0.0, undesirable_weight: 0.0, ..CFG }),
            ("seq_len", FeedbackConfig { seq_len: 1, ..CFG }),
        ] {
            let why = cfg.validate().expect_err(field);
            assert!(why.contains(field.split(' ').next().unwrap()) || field == "both zero", "{field}: {why}");
        }
        let one = PackedFeedback::from_masked(8, (&[1, 2, 3], &[false, true, true]), (&[1, 2, 4], &[false, true, true]), true).unwrap();
        assert!(matches!(Feedback::try_from_dataset(CFG, vec![one.clone()]), Err(FeedbackConfigError::TooFewExamples(1))));
        let narrow = PackedFeedback::from_masked(4, (&[1, 2, 3], &[false, true, true]), (&[1, 2, 4], &[false, true, true]), false).unwrap();
        assert!(matches!(Feedback::try_from_dataset(CFG, vec![one, narrow]), Err(FeedbackConfigError::Width { index: 1, got: 4, expected: 8 })));
        assert!(PackedFeedback::from_masked(2, (&[1, 2, 3], &[false, true, true]), (&[1, 2], &[false, true]), true).unwrap_err().contains("do not fit"));
        assert!(PackedFeedback::from_masked(8, (&[1, 2, 3], &[false, false, false]), (&[1, 2], &[false, true]), true).unwrap_err().contains("no supervised token"));
    }
}

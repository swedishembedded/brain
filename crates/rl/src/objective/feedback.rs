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
//! ## One batch, two rows
//!
//! An example is packed as the two rows of one `b = 2` batch: row 0 its own completion (trained), row 1 its prompt
//! with the next example's completion (forwarded for `z`, weight zero). The reference is the model the run starts
//! from, scored once per row before the first step ([`Objective::check_data`]), so no second model is held. A step
//! of `k` micro-batches forwards every one of its examples once to measure `z`, then again to train.

use data::rng::Rng;
use model::training::Unsupported;
use model::{Batch, Model, ModelConfig, Objective, StepVerdict, IGNORE};

use super::dpo::row_sum;

/// The default `beta` (the paper's).
pub const DEFAULT_KTO_BETA: f32 = 0.1;
/// The default weights of the desirable and of the undesirable examples' terms.
pub const DEFAULT_KTO_DESIRABLE_WEIGHT: f32 = 1.0;
pub const DEFAULT_KTO_UNDESIRABLE_WEIGHT: f32 = 1.0;

/// KTO's hyperparameters. `seq_len` is one row's length; the model is built at `b = 2`, `t = seq_len`.
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
/// forward of `count` active positions (both rows). Evaluated in f64.
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

/// One example packed as the two rows of a `b = 2` batch: its own completion (row 0) and its prompt with another
/// example's completion (row 1), with the frozen reference's log-probability of every supervised target.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedFeedback {
    tokens: Vec<u32>,
    targets: Vec<u32>,
    ref_lp: Vec<f32>,
    desirable: bool,
}

impl PackedFeedback {
    /// Pack two rendered conversations (token ids and a supervision mask each): `own` the example, `mismatched` its
    /// prompt followed by another example's completion. Position `t` predicts token `t + 1` and is a target exactly
    /// when that token is supervised. Refused, never truncated: a conversation longer than `seq_len`, a mask of another
    /// length than its ids, a row with no supervised token after the first.
    pub fn from_masked(seq_len: usize, own: (&[u32], &[bool]), mismatched: (&[u32], &[bool]), desirable: bool) -> Result<PackedFeedback, String> {
        let mut tokens = vec![0u32; 2 * seq_len];
        let mut targets = vec![IGNORE; 2 * seq_len];
        for (row, (name, (ids, mask))) in [("completion", own), ("mismatched completion", mismatched)].into_iter().enumerate() {
            if ids.len() != mask.len() {
                return Err(format!("{name}: {} token ids but {} mask entries", ids.len(), mask.len()));
            }
            if ids.len() > seq_len {
                return Err(format!("{name}: {} tokens do not fit a row of {seq_len}", ids.len()));
            }
            let base = row * seq_len;
            tokens[base..base + ids.len()].copy_from_slice(ids);
            for t in 1..ids.len() {
                if mask[t] {
                    targets[base + t - 1] = ids[t];
                }
            }
            if targets[base..base + seq_len].iter().all(|&y| y == IGNORE) {
                return Err(format!("{name}: no supervised token after the first, so the row has no log-probability"));
            }
        }
        Ok(PackedFeedback { tokens, targets, ref_lp: vec![0f32; 2 * seq_len], desirable })
    }

    /// Make `reference` the frozen reference of both rows: one forward, keeping its log-probability of every target.
    pub fn score_reference<M: Model>(&mut self, reference: &M) {
        reference.set_batch(Batch::Lm { tokens: &self.tokens, targets: &self.targets });
        let _ = reference.forward();
        let lp = model::training::objectives(reference).batch_token_logprobs();
        for (i, &y) in self.targets.iter().enumerate() {
            self.ref_lp[i] = if y == IGNORE { 0.0 } else { lp[i] };
        }
    }

    pub fn desirable(&self) -> bool {
        self.desirable
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn targets(&self) -> &[u32] {
        &self.targets
    }

    fn seq_len(&self) -> usize {
        self.tokens.len() / 2
    }

    /// Refuse a token id the model has no embedding row for.
    pub fn check_vocab(&self, vocab: u32) -> Result<(), String> {
        match self.tokens.iter().copied().chain(self.targets.iter().copied().filter(|&y| y != IGNORE)).find(|&id| id >= vocab) {
            Some(id) => Err(format!("token id {id} is outside the model's vocabulary of {vocab}")),
            None => Ok(()),
        }
    }
}

/// The log-ratios of both rows of `packed` under `model` (forwarded here): its own and the mismatched one.
pub fn logratios<M: Model>(model: &M, packed: &PackedFeedback) -> (f32, f32) {
    let seq_len = packed.seq_len();
    model.set_batch(Batch::Lm { tokens: &packed.tokens, targets: &packed.targets });
    let _ = model.forward();
    let new_lp = model::training::objectives(model).batch_token_logprobs();
    let (new_own, ref_own, _) = row_sum(&new_lp, &packed.ref_lp, &packed.targets, 0, seq_len);
    let (new_mis, ref_mis, _) = row_sum(&new_lp, &packed.ref_lp, &packed.targets, 1, seq_len);
    (new_own - ref_own, new_mis - ref_mis)
}

/// One example's training forward against the held reference point `z`: the term, and its weight installed on the
/// supervised positions of row 0 (row 1 weighs nothing) - everything a micro-step does before its backward. Also
/// returns the example's log-ratio. `model` is built at `b = 2`, `t = seq_len`, with weighted loss enabled.
pub fn weigh_feedback<M: Model>(cfg: &FeedbackConfig, model: &M, packed: &PackedFeedback, z: f32) -> (FeedbackTerm, f32) {
    let seq_len = packed.seq_len();
    model.set_batch(Batch::Lm { tokens: &packed.tokens, targets: &packed.targets });
    let _ = model.forward();
    let objectives = model::training::objectives(model);
    let new_lp = objectives.batch_token_logprobs();
    let (new_own, ref_own, count_own) = row_sum(&new_lp, &packed.ref_lp, &packed.targets, 0, seq_len);
    let (_, _, count_mis) = row_sum(&new_lp, &packed.ref_lp, &packed.targets, 1, seq_len);
    let logratio = new_own - ref_own;
    let term = feedback_term(cfg, packed.desirable, logratio, z, (count_own + count_mis).max(1) as f32);
    let mut weights = vec![0f32; 2 * seq_len];
    for (i, w) in weights.iter_mut().enumerate().take(seq_len) {
        if packed.targets[i] != IGNORE {
            *w = term.weight;
        }
    }
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
    referenced: bool,
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
        Ok(Feedback { cfg, examples, referenced: false, vocab: None, trained: 0, last_z: 0.0, last_logratio: 0.0 })
    }
}

impl Feedback {
    /// Each example's log-ratio of its own completion under `model` against the reference it was scored with.
    pub fn logratios<M: Model>(&self, model: &M) -> Vec<f32> {
        self.examples.iter().map(|e| logratios(model, e).0).collect()
    }
}

impl<M: Model> Objective<M> for Feedback {
    fn regime(&self) -> &'static str {
        "feedback"
    }

    fn prepare(&mut self, model: &mut M) -> Result<(), Unsupported> {
        model::training::require_token_objectives(model)?.enable_weighted_loss();
        Ok(())
    }

    fn check_data(&mut self, model: &M) -> Result<(), std::io::Error> {
        let vocab = model.config().vocab();
        self.vocab = Some(vocab);
        for (index, e) in self.examples.iter().enumerate() {
            e.check_vocab(vocab).map_err(|why| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("feedback example {index}: {why}")))?;
        }
        if !self.referenced {
            for e in &mut self.examples {
                e.score_reference(model);
            }
            self.referenced = true;
        }
        Ok(())
    }

    fn micro_step(&mut self, model: &M, rng: &mut Rng) -> f32 {
        Objective::<M>::micro_steps(self, model, rng, 1)
    }

    /// Draw the step's examples, measure the reference point on their mismatched rows, then train each against it.
    fn micro_steps(&mut self, model: &M, rng: &mut Rng, k: u32) -> f32 {
        assert!(self.referenced, "Feedback: the reference is scored in check_data(), which was not called");
        let drawn: Vec<usize> = (0..k.max(1)).map(|_| rng.gen_range_inclusive(0, self.examples.len() as i64 - 1) as usize).collect();
        let mismatched: Vec<f32> = drawn.iter().map(|&e| logratios(model, &self.examples[e]).1).collect();
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
        let z = reference_point(&[logratios(model, &self.examples[e]).1]);
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

    /// Both rows of an example share one forward.
    fn batch_shape(&self) -> Option<(u32, u32)> {
        Some((2, self.cfg.seq_len as u32))
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

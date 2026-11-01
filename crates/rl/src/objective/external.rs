// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training language models from the rollouts an agent collected, for its
// clients. If your team needs expertise in reinforcement learning from externally collected agent trajectories then you
// can procure our services by sending an email to info@swedishembedded.com.

//! Policy-gradient training on rollouts collected outside brain.
//!
//! An orchestrator runs an agent, turns every model call into a chat record (the assistant turn of the call is the
//! action), scores each rollout with an independent verifier and writes the group-relative advantage on each of the
//! rollout's records ([`data::chat_rollout`]). [`ExternalPolicyGradient`] trains a LoRA adapter on such a prepared
//! dataset with the clipped policy-gradient objective of GRPO/PPO on the supervised (action) tokens: records with a
//! negative advantage push their actions' probability down, those with a positive one pull it up. The per-token term
//! and the k3 KL are the functions [`super::grpo::Grpo`] trains through ([`super::pg_batch`]); only the source of the
//! rows differs.
//!
//! ## The objective
//!
//! For a supervised token with advantage `A` (of its record), behaviour log-probability `old` and current
//! log-probability `new`, the loss is `-min(r A, clip(r, 1 - eps, 1 + eps) A) + beta * k3(ref, new)` with
//! `r = exp(new - old)`. A token's gradient is zero where the clip is active on the side the advantage pushes toward.
//!
//! * **`old`, the behaviour policy.** The record's `behavior_logprobs` when the collector recorded them (the
//!   log-probabilities of `softmax(logits / temperature)` the sampler drew with). Where it recorded none, `old` is the
//!   log-probability of the adapter the run **started from**, measured by one forward pass over the dataset before the
//!   first step (a fresh adapter: the base). At the first optimiser step `r = 1` and the gradient is the plain
//!   on-policy estimator `-A * grad(new)`; the clip then limits how far later steps, and later epochs over the same
//!   rows, may move a token from where the run began. That is a proxy for the true behaviour policy, which is
//!   unknown, and is exact when the rollouts were sampled from the adapter the run starts from.
//! * **`ref`, the reference policy** ([`ReferencePolicy`]): the base model (the adapter disabled), or the adapter the
//!   run started from. They are the same for a fresh adapter. Measured by a forward pass over the dataset before the
//!   first step; the KL term is the k3 estimator `e^u - u - 1`, `u = ref - new`, at the sampled tokens. Its mean is
//!   reported even when `kl_beta` is zero ([`PolicyGradientConfig::log_kl`]).
//! * **Normalisation** ([`LossAgg`]): `SeqMean` divides each sequence's token terms by its own supervised-token
//!   count (GRPO as published: needs a batch of one sequence); `TokenSum` divides every token of an optimiser step by
//!   the step's mean count per micro-step, so all tokens weigh the same.
//! * **Records without an advantage** are imitation examples: plain cross-entropy (weight 1, no KL), so a dataset may
//!   mix rollouts with supervised records.
//!
//! The row of a record with images is read through the model's shared vision path (`Batch::LmExamples`), so images
//! train as they do in supervised fine-tuning: the tower is frozen and the loss covers the action tokens only.

use std::path::Path;

use data::binio;
use data::loader::{BatchConfig, TokenDataset};
use data::rng::Rng;
use model::training::{self, Capability, Unsupported};
use model::{Batch, FitOpts, Model, Objective, Split, StepVerdict, IGNORE};

use super::grpo::LossAgg;
use super::grpo_stats::{NonFiniteSteps, MAX_CONSECUTIVE_NONFINITE};
use super::grpo_term::ClipBand;
use super::pg_batch::{adapters_are_zero, backward_weighted, weigh, with_adapters_disabled, BatchTerms, PgBatch, PgSpec};

/// The policy the KL term keeps the trained policy close to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferencePolicy {
    /// The frozen base model: the LoRA adapters disabled. Needs a model with adapters.
    #[default]
    Base,
    /// The policy the run starts from: the adapter being continued, or the base for a fresh adapter.
    StartOfRun,
}

impl ReferencePolicy {
    /// The name the CLI and the adapter's provenance use.
    pub fn name(self) -> &'static str {
        match self {
            ReferencePolicy::Base => "base",
            ReferencePolicy::StartOfRun => "start",
        }
    }
}

impl std::str::FromStr for ReferencePolicy {
    type Err = String;
    fn from_str(s: &str) -> Result<ReferencePolicy, String> {
        match s {
            "base" => Ok(ReferencePolicy::Base),
            "start" => Ok(ReferencePolicy::StartOfRun),
            other => Err(format!("unknown reference policy {other:?}; expected \"base\" or \"start\"")),
        }
    }
}

/// The knobs of [`ExternalPolicyGradient`]. [`Default`] is the recommended configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicyGradientConfig {
    /// Half-width of the clip band `[1 - eps, 1 + eps]` of the importance ratio, in `(0, 1)`.
    pub clip_eps: f32,
    /// Coefficient of the k3 KL to the reference policy; `0.0` puts no KL term in the loss.
    pub kl_beta: f32,
    pub reference: ReferencePolicy,
    /// How token terms are normalised ([`LossAgg::SeqMean`], as GRPO publishes it, by default).
    pub loss_agg: LossAgg,
    /// The temperature the collector sampled with; the policy trained is `softmax(logits / temperature)`.
    pub temperature: f32,
    /// Measure and report the KL to the reference even when `kl_beta` is zero (one more forward pass over the data).
    pub log_kl: bool,
}

impl Default for PolicyGradientConfig {
    fn default() -> PolicyGradientConfig {
        PolicyGradientConfig { clip_eps: 0.2, kl_beta: 0.02, reference: ReferencePolicy::Base, loss_agg: LossAgg::SeqMean, temperature: 1.0, log_kl: true }
    }
}

impl PolicyGradientConfig {
    /// Refuse a setting that has no meaning.
    pub fn validate(&self) -> Result<(), String> {
        if !(self.clip_eps.is_finite() && self.clip_eps > 0.0 && self.clip_eps < 1.0) {
            return Err(format!("clip_eps is {}; it must lie in (0, 1)", self.clip_eps));
        }
        if !(self.kl_beta.is_finite() && self.kl_beta >= 0.0) {
            return Err(format!("kl_beta is {}; it must be a finite number, at least 0", self.kl_beta));
        }
        if !(self.temperature.is_finite() && self.temperature > 0.0) {
            return Err(format!("temperature is {}; it must be finite and positive", self.temperature));
        }
        Ok(())
    }

    fn wants_reference(&self) -> bool {
        self.kl_beta > 0.0 || self.log_kl
    }
}

/// What one optimiser step measured. Losses are means per supervised token.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct PolicyGradientStep {
    /// 1-based index of the optimiser step among those the objective ran.
    pub step: usize,
    /// Sequences trained and supervised tokens in the step.
    pub sequences: usize,
    pub tokens: usize,
    /// The mean advantage-weighted loss: the clipped surrogate (negated), the KL term and the cross-entropy of
    /// imitation tokens, per supervised token.
    pub loss: f32,
    /// Of the tokens with a non-zero advantage, the fraction whose ratio was outside the clip band on the side the
    /// advantage pushes toward (no gradient).
    pub clip_fraction: f32,
    /// Mean k3 KL to the reference over the policy tokens, when a reference was measured.
    pub kl: Option<f32>,
    /// The importance ratio over the policy tokens: mean, smallest and largest.
    pub ratio_mean: f32,
    pub ratio_min: f32,
    pub ratio_max: f32,
    /// Mean advantage of the step's sequences (imitation sequences excluded).
    pub mean_advantage: f32,
    /// Tokens trained as imitation (a record without an advantage).
    pub imitation_tokens: usize,
}

/// Everything a run measured, per optimiser step, and how it began.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct PolicyGradientStats {
    pub steps: Vec<PolicyGradientStep>,
    /// Optimiser steps not taken because a loss or gradient was not finite.
    pub steps_skipped_nonfinite: usize,
    /// Sequences of the dataset with an advantage, and without (imitation).
    pub sequences_with_advantage: usize,
    pub sequences_imitation: usize,
    /// Supervised tokens whose behaviour log-probability was recorded, and those that took the start-of-run value.
    pub tokens_recorded_behavior: usize,
    pub tokens_start_of_run_behavior: usize,
}

impl PolicyGradientStats {
    /// The mean over steps of a per-step quantity, `None` before the first step.
    fn mean(&self, f: impl Fn(&PolicyGradientStep) -> f32) -> Option<f32> {
        (!self.steps.is_empty()).then(|| self.steps.iter().map(&f).sum::<f32>() / self.steps.len() as f32)
    }

    /// The mean advantage-weighted loss over all steps.
    pub fn mean_loss(&self) -> Option<f32> {
        self.mean(|s| s.loss)
    }

    /// The mean clip fraction over all steps.
    pub fn mean_clip_fraction(&self) -> Option<f32> {
        self.mean(|s| s.clip_fraction)
    }

    /// The mean KL to the reference over the steps that measured one.
    pub fn mean_kl(&self) -> Option<f32> {
        let kls: Vec<f32> = self.steps.iter().filter_map(|s| s.kl).collect();
        (!kls.is_empty()).then(|| kls.iter().sum::<f32>() / kls.len() as f32)
    }
}

/// The rollout data of the training split: where each example starts and what the collector recorded.
struct Rows {
    starts: Vec<usize>,
    ends: Vec<usize>,
    advantage: Vec<f32>,
    /// Per token of the stream; NaN where none was recorded.
    behavior: Vec<f32>,
    /// Per example, per row position: the log-probability of the policy the run started from, and of the reference.
    /// Empty until [`ExternalPolicyGradient::prepare`] measured them, and for examples that need neither.
    start_lp: Vec<Vec<f32>>,
    reference_lp: Vec<Vec<f32>>,
}

/// The order examples are drawn in: every example once per epoch, reshuffled each epoch from the seed alone, so a run
/// resumed at draw `n` continues the order a straight run follows.
struct Order {
    seed: u64,
    examples: usize,
    drawn: u64,
    epoch: u64,
    permutation: Vec<usize>,
}

impl Order {
    fn new(seed: u64, examples: usize, drawn: u64) -> Order {
        let epoch = drawn / examples as u64;
        Order { seed, examples, drawn, epoch, permutation: model::train::epoch_order(seed, examples, epoch) }
    }

    fn next(&mut self) -> usize {
        let epoch = self.drawn / self.examples as u64;
        if epoch != self.epoch {
            self.epoch = epoch;
            self.permutation = model::train::epoch_order(self.seed, self.examples, epoch);
        }
        let pick = self.permutation[(self.drawn % self.examples as u64) as usize];
        self.drawn += 1;
        pick
    }
}

/// One micro-batch laid out for the model.
struct Laid {
    tokens: Vec<u32>,
    targets: Vec<u32>,
    picks: Vec<Option<usize>>,
    old_lp: Vec<f32>,
    ref_lp: Vec<f32>,
    advantage: Vec<f32>,
    has_ref: bool,
}

/// The step being accumulated.
#[derive(Default)]
struct StepAccumulator {
    terms: Vec<BatchTerms>,
    sequences: usize,
    advantage_sum: f64,
    advantaged_sequences: usize,
}

/// Clipped policy-gradient training on a prepared dataset of externally collected rollouts. See the module docs.
pub struct ExternalPolicyGradient {
    cfg: PolicyGradientConfig,
    train: TokenDataset,
    batch_cfg: BatchConfig,
    rows: Rows,
    /// The dataset holds images: batches name their examples so the model reads them.
    tag_examples: bool,
    order: Option<Order>,
    accumulating: StepAccumulator,
    stats: PolicyGradientStats,
    consecutive_nonfinite: u32,
    verdict: Option<StepVerdict>,
}

impl ExternalPolicyGradient {
    /// The objective over the prepared dataset in `dir` (written by `data::chat::prepare_chat_samples` from records
    /// that may carry an advantage). `opts` sizes the batches as the supervised fine-tune sizes them.
    ///
    /// # Errors
    /// The configuration is invalid; the dataset has no examples or its rollout sidecars disagree with it; a
    /// sequence-mean objective is asked to train a batch of more than one sequence.
    pub fn open(dir: &Path, opts: &FitOpts, cfg: PolicyGradientConfig) -> std::io::Result<ExternalPolicyGradient> {
        let invalid = |why: String| std::io::Error::new(std::io::ErrorKind::InvalidInput, why);
        cfg.validate().map_err(invalid)?;
        if cfg.loss_agg == LossAgg::SeqMean && opts.batch_size != 1 {
            return Err(invalid(format!("a sequence-mean objective trains one sequence per batch row group, the batch holds {}; use loss_agg TokenSum or a batch of 1", opts.batch_size)));
        }
        let (train, _val, batch_cfg, _vocab, _itos) = model::load_rollout_dataset(dir, opts)?;
        let tokens = binio::read_tokens_u32(&dir.join("train"))?;
        let starts: Vec<usize> = binio::read_u64_bin(&dir.join("train.ex.bin"))?.into_iter().map(|s| s as usize).collect();
        if starts.is_empty() {
            return Err(invalid(format!("{}: the training split holds no examples", dir.display())));
        }
        let ends: Vec<usize> = (0..starts.len()).map(|i| starts.get(i + 1).copied().unwrap_or(tokens.len())).collect();
        let mask = binio::read_mask_bin(&dir.join("train.mask.bin"))?;
        let recorded = data::chat_rollout::read(dir, "train", starts.len(), tokens.len())?;
        let (advantage, behavior) = match recorded {
            Some(r) => (r.advantage, r.behavior),
            None => (vec![f32::NAN; starts.len()], vec![f32::NAN; tokens.len()]),
        };
        let mut stats = PolicyGradientStats::default();
        for (e, &a) in advantage.iter().enumerate() {
            if a.is_nan() {
                stats.sequences_imitation += 1;
                continue;
            }
            stats.sequences_with_advantage += 1;
            // Row position t predicts token start + 1 + t.
            for token in (starts[e] + 1)..ends[e] {
                if mask[token] {
                    if behavior[token].is_nan() {
                        stats.tokens_start_of_run_behavior += 1;
                    } else {
                        stats.tokens_recorded_behavior += 1;
                    }
                }
            }
        }
        let tag_examples = ["train", "val"].iter().any(|split| dir.join(format!("{split}.img.bin")).exists());
        let rows = Rows { starts, ends, advantage, behavior, start_lp: Vec::new(), reference_lp: Vec::new() };
        Ok(ExternalPolicyGradient { cfg, train, batch_cfg, rows, tag_examples, order: None, accumulating: StepAccumulator::default(), stats, consecutive_nonfinite: 0, verdict: None })
    }

    /// What the run has measured so far.
    pub fn stats(&self) -> &PolicyGradientStats {
        &self.stats
    }

    /// [`Self::stats`], consuming the objective.
    pub fn into_stats(self) -> PolicyGradientStats {
        self.stats
    }

    fn spec(&self) -> PgSpec {
        PgSpec { band: ClipBand::symmetric(self.cfg.clip_eps), kl_beta: self.cfg.kl_beta }
    }

    /// Whether some supervised token of an example with an advantage has no recorded behaviour log-probability.
    fn needs_start_of_run_behavior(&self) -> bool {
        self.stats.tokens_start_of_run_behavior > 0
    }

    fn upload<M: Model>(&self, model: &M, tokens: &[u32], targets: &[u32], picks: &[Option<usize>]) {
        if self.tag_examples {
            model.set_batch(Batch::LmExamples { split: Split::Train, tokens, targets, examples: picks });
        } else {
            model.set_batch(Batch::Lm { tokens, targets });
        }
    }

    /// The batch whose rows are the examples `picks` names, with the per-position inputs of the objective.
    fn lay_out(&self, picks: &[Option<usize>]) -> Laid {
        let (x, y) = self.train.example_batch(&self.batch_cfg, picks).expect("a prepared chat dataset indexes its examples");
        let targets: Vec<u32> = y.iter().map(|&v| if v < 0 { IGNORE } else { v as u32 }).collect();
        let (bs, bl) = (self.batch_cfg.batch_size, self.batch_cfg.block_size);
        let has_ref = !self.rows.reference_lp.is_empty();
        let mut laid = Laid { tokens: x, targets, picks: picks.to_vec(), old_lp: vec![0.0; bs * bl], ref_lp: vec![0.0; bs * bl], advantage: vec![f32::NAN; bs * bl], has_ref };
        for (b, pick) in picks.iter().enumerate() {
            let Some(e) = *pick else { continue };
            for t in 0..bl.min(self.rows.ends[e] - self.rows.starts[e]) {
                let at = b * bl + t;
                if laid.targets[at] == IGNORE {
                    continue;
                }
                laid.advantage[at] = self.rows.advantage[e];
                let recorded = self.rows.behavior[self.rows.starts[e] + 1 + t];
                laid.old_lp[at] = if recorded.is_nan() { self.rows.start_lp.get(e).and_then(|lp| lp.get(t)).copied().unwrap_or(0.0) } else { recorded };
                if has_ref {
                    laid.ref_lp[at] = self.rows.reference_lp[e].get(t).copied().unwrap_or(0.0);
                }
            }
        }
        laid
    }

    /// The model's log-probability of every supervised token of every example, by one forward pass per batch.
    fn measure_policy<M: Model>(&self, model: &M) -> Vec<Vec<f32>> {
        let (bs, bl) = (self.batch_cfg.batch_size, self.batch_cfg.block_size);
        let examples = self.rows.starts.len();
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(examples);
        for first in (0..examples).step_by(bs) {
            let picks: Vec<Option<usize>> = (first..first + bs).map(|e| (e < examples).then_some(e)).collect();
            let laid = self.lay_out(&picks);
            self.upload(model, &laid.tokens, &laid.targets, &laid.picks);
            let _ = model.forward();
            let lp = training::objectives(model).batch_token_logprobs();
            for (b, pick) in picks.iter().enumerate() {
                if let Some(e) = pick {
                    let len = (self.rows.ends[*e] - self.rows.starts[*e]).min(bl);
                    out.push(lp[b * bl..b * bl + len].to_vec());
                }
            }
        }
        out
    }

    /// The step's statistics from what its micro-batches measured.
    fn close_step(&mut self) {
        let acc = std::mem::take(&mut self.accumulating);
        let mut sum = BatchTerms::default();
        for t in &acc.terms {
            sum.loss += t.loss;
            sum.supervised += t.supervised;
            sum.policy_tokens += t.policy_tokens;
            sum.clipped += t.clipped;
            sum.signed_tokens += t.signed_tokens;
            sum.imitation_tokens += t.imitation_tokens;
            sum.kl_sum += t.kl_sum;
            sum.kl_tokens += t.kl_tokens;
            sum.ratio_sum += t.ratio_sum;
            sum.ratio_min = sum.ratio_min.min(t.ratio_min);
            sum.ratio_max = sum.ratio_max.max(t.ratio_max);
        }
        let per_token = |x: f32| if sum.supervised > 0 { x / sum.supervised as f32 } else { 0.0 };
        let policy = sum.policy_tokens.max(1) as f32;
        self.stats.steps.push(PolicyGradientStep {
            step: self.stats.steps.len() + 1,
            sequences: acc.sequences,
            tokens: sum.supervised,
            loss: per_token(sum.loss),
            clip_fraction: if sum.signed_tokens > 0 { sum.clipped as f32 / sum.signed_tokens as f32 } else { 0.0 },
            kl: (sum.kl_tokens > 0).then(|| sum.kl_sum / sum.kl_tokens as f32),
            ratio_mean: if sum.policy_tokens > 0 { (sum.ratio_sum / f64::from(policy)) as f32 } else { 1.0 },
            ratio_min: if sum.policy_tokens > 0 { sum.ratio_min } else { 1.0 },
            ratio_max: if sum.policy_tokens > 0 { sum.ratio_max } else { 1.0 },
            mean_advantage: if acc.advantaged_sequences > 0 { (acc.advantage_sum / acc.advantaged_sequences as f64) as f32 } else { 0.0 },
            imitation_tokens: sum.imitation_tokens,
        });
    }

    fn rule(&mut self, finite: bool) -> StepVerdict {
        if finite {
            self.consecutive_nonfinite = 0;
            return StepVerdict::Apply;
        }
        self.consecutive_nonfinite += 1;
        self.stats.steps_skipped_nonfinite += 1;
        if self.consecutive_nonfinite >= MAX_CONSECUTIVE_NONFINITE {
            StepVerdict::Abort(std::io::Error::new(std::io::ErrorKind::InvalidData, NonFiniteSteps { consecutive: self.consecutive_nonfinite }))
        } else {
            StepVerdict::Skip(format!("non-finite loss or gradient ({} in a row)", self.consecutive_nonfinite))
        }
    }

    fn draw(&mut self) -> Vec<Option<usize>> {
        let order = self.order.as_mut().expect("Objective::begin positions the order before the first step");
        (0..self.batch_cfg.batch_size).map(|_| Some(order.next())).collect()
    }
}

impl<M: Model> Objective<M> for ExternalPolicyGradient {
    fn regime(&self) -> &'static str {
        "policy_gradient_external"
    }

    /// Opt the model into weighted-loss training, set the sampling temperature, and measure the two anchor policies
    /// (the behaviour proxy and the reference) the dataset needs, before any step and before a resumed run restores
    /// its weights: both are the policy the run was started from.
    fn prepare(&mut self, model: &mut M) -> Result<(), Unsupported> {
        let objectives = training::require_token_objectives(model)?;
        objectives.enable_weighted_loss();
        objectives.set_logit_scale(1.0 / self.cfg.temperature);
        let wants_reference = self.cfg.wants_reference() && self.stats.sequences_with_advantage > 0;
        let wants_start = self.needs_start_of_run_behavior() || (wants_reference && self.cfg.reference == ReferencePolicy::StartOfRun);
        let zero_adapter = adapters_are_zero(model);
        let start = (wants_start || (wants_reference && zero_adapter)).then(|| self.measure_policy(model));
        if wants_reference {
            self.rows.reference_lp = match (self.cfg.reference, start.as_ref()) {
                (ReferencePolicy::StartOfRun, Some(start)) => start.clone(),
                // A fresh adapter is the base: the pass already made is the reference.
                (ReferencePolicy::Base, Some(start)) if zero_adapter => start.clone(),
                _ => with_adapters_disabled(model, || self.measure_policy(model)).ok_or_else(|| Unsupported::of::<M>(Capability::ChatLoraFineTune))?,
            };
        }
        if let Some(start) = start.filter(|_| self.needs_start_of_run_behavior()) {
            self.rows.start_lp = start;
        }
        Ok(())
    }

    fn begin(&mut self, seed: u64, draws_done: u64) {
        self.order = Some(Order::new(seed, self.rows.starts.len(), draws_done));
    }

    fn micro_step(&mut self, model: &M, rng: &mut Rng) -> f32 {
        self.micro_steps(model, rng, 1)
    }

    /// `k` micro-batches of one optimiser step. Their gradients accumulate; a model with a loss denominator divides
    /// each by the step's mean supervised-token count under [`LossAgg::TokenSum`].
    fn micro_steps(&mut self, model: &M, _rng: &mut Rng, k: u32) -> f32 {
        let laid: Vec<Laid> = (0..k.max(1)).map(|_| self.draw()).collect::<Vec<_>>().iter().map(|picks| self.lay_out(picks)).collect();
        let supervised: usize = laid.iter().map(|l| l.targets.iter().filter(|&&t| t != IGNORE).count()).sum();
        let denominator = match self.cfg.loss_agg {
            LossAgg::TokenSum if supervised > 0 => Some(supervised as f32 / laid.len() as f32),
            _ => None,
        };
        let spec = self.spec();
        let mut loss = 0.0;
        let mut finite = true;
        for l in &laid {
            self.upload(model, &l.tokens, &l.targets, &l.picks);
            let _ = model.forward();
            let batch = PgBatch { targets: &l.targets, old_lp: &l.old_lp, ref_lp: l.has_ref.then_some(&l.ref_lp[..]), advantage: &l.advantage };
            let terms = backward_weighted(model, &batch, &spec, denominator);
            finite &= terms.finite;
            loss += terms.loss / denominator.unwrap_or(terms.supervised.max(1) as f32);
            for e in l.picks.iter().flatten() {
                self.accumulating.sequences += 1;
                if !self.rows.advantage[*e].is_nan() {
                    self.accumulating.advantaged_sequences += 1;
                    self.accumulating.advantage_sum += f64::from(self.rows.advantage[*e]);
                }
            }
            self.accumulating.terms.push(terms);
        }
        finite &= model.optimized_params().is_none_or(|names| names.iter().all(|n| model.read_grad(n).iter().all(|g| g.is_finite())));
        self.verdict = Some(self.rule(finite));
        self.close_step();
        loss
    }

    fn step_verdict(&mut self) -> StepVerdict {
        self.verdict.take().unwrap_or(StepVerdict::Apply)
    }

    /// The loss of freshly drawn examples on the forward alone: nothing is queued, counted or differentiated.
    fn loss_probe(&mut self, model: &M, rng: &mut Rng) -> f32 {
        let examples = self.rows.starts.len();
        let picks: Vec<Option<usize>> = (0..self.batch_cfg.batch_size).map(|_| Some(rng.gen_range_inclusive(0, examples as i64 - 1) as usize)).collect();
        let laid = self.lay_out(&picks);
        self.upload(model, &laid.tokens, &laid.targets, &laid.picks);
        let _ = model.forward();
        let batch = PgBatch { targets: &laid.targets, old_lp: &laid.old_lp, ref_lp: laid.has_ref.then_some(&laid.ref_lp[..]), advantage: &laid.advantage };
        let (_, terms) = weigh(&training::objectives(model).batch_token_logprobs(), &batch, &self.spec());
        terms.loss / terms.supervised.max(1) as f32
    }

    fn metrics(&self) -> Vec<(&'static str, f32)> {
        let Some(last) = self.stats.steps.last() else { return Vec::new() };
        let mut m = vec![("pg_loss", last.loss), ("pg_clip_fraction", last.clip_fraction), ("pg_ratio_mean", last.ratio_mean), ("pg_ratio_min", last.ratio_min), ("pg_ratio_max", last.ratio_max), ("pg_mean_advantage", last.mean_advantage)];
        if let Some(kl) = last.kl {
            m.push(("pg_kl", kl));
        }
        m
    }
}

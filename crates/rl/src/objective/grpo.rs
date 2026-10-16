// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GRPO: group-relative, clipped-importance-ratio policy-gradient training,
//! with an optional k3 KL-to-a-frozen-reference term - and plain
//! rejection-sampling/STaR training as the SAME weight formula's degenerate
//! (single-completion) case, not a parallel implementation next to it.
//!
//! Swedish Embedded AB builds the training-regime machinery that turns a
//! verifiable-reward environment into a trained model - GRPO, rejection
//! sampling, and everything in between reduce to one gradient-checked
//! weighted cross-entropy kernel here. If your team needs RL-from-verifiable-
//! rewards on top of a from-scratch training stack, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! ## The reduction to a per-token weight
//!
//! See [`super::grpo_term`]: one ordinary forward reads the current policy's
//! per-token log-probabilities, the host computes each token's clipped
//! surrogate weight, [`model::training::TokenObjectives::set_loss_weights`]
//! installs them, and one ordinary backward differentiates the objective.
//!
//! ## One optimiser step
//!
//! [`Objective::micro_steps`] is one optimiser step's worth of rows. It gathers
//! `k` rows with a non-zero advantage, drawing further groups while it is short
//! (dynamic sampling, bounded by [`GrpoOptions::resample_factor`]), trains them
//! with one shared loss denominator ([`LossAgg`]), and rules through
//! [`Objective::step_verdict`] whether the step may be taken: never on zero
//! rows, never on a non-finite loss or gradient.
//!
//! ## The distribution that is trained is the one that sampled
//!
//! A sampler at temperature `T` draws from `softmax(logits / T)`.
//! [`Objective::prepare`] sets the model's logit scale to `1/T`, so the
//! trainer's log-probabilities are under that same distribution; the model's
//! own rollout then samples at temperature 1 over the already scaled logits.
//! The behaviour log-probability of every row is recomputed by the training
//! forward when the row is packed, so a different kernel or device in the
//! sampler never enters the ratio; the sampler's own value is kept only to log
//! how far it was from the trainer's ([`GrpoStats::max_sampler_gap`]).

use data::rng::Rng;
use model::rollout::{Completion, ModelRollout, Rollout, RolloutParams, StopReason};
use model::training::{self, Unsupported};
use model::{Batch, Model, Objective, StepVerdict, IGNORE};

pub use super::advantage::{group_advantages, group_advantages_with, AdvantageNorm};
pub use super::grpo_stats::{CycleLog, GrpoStats, NonFiniteSteps, RowRefusal, MAX_CONSECUTIVE_NONFINITE};
pub use super::grpo_term::{k3_kl, near_clip_boundary, token_term, token_term_banded, ClipBand, TokenTerm, MAX_LOG_RATIO};
use crate::env::{Environment, Task, Verifier};

/// How a row's per-token loss terms are normalised.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LossAgg {
    /// Each row's terms are divided by that row's own supervised-token count
    /// (the legacy behaviour): a token in a long completion weighs less than
    /// one in a short completion, so length is rewarded or punished by the
    /// normalisation alone.
    SeqMean,
    /// Every token of the optimiser step is divided by the step's mean
    /// supervised-token count per micro-step: all tokens weigh the same
    /// whatever the length of the row they sit in.
    #[default]
    TokenSum,
}

/// Which distribution the importance ratio is taken in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RatioBasis {
    /// The trained distribution is the sampled one: the model's logit scale is
    /// `1/T` and the behaviour log-probabilities are recomputed by the
    /// training forward.
    #[default]
    Trainer,
    /// The legacy basis, for ablations only: the trainer scores the untempered
    /// softmax while the sampler drew at temperature `T`, and the behaviour
    /// log-probability is the sampler's own. At `T != 1` every on-policy ratio
    /// starts off 1 (see [`GrpoStats::max_onpolicy_ratio_dev`]), and a confident
    /// token with a positive advantage starts clipped.
    Sampler,
}

/// The knobs beyond the classic GRPO set. [`Default`] is the recommended
/// configuration; the legacy behaviour is `advantage: ZScore`, `loss_agg:
/// SeqMean` and `clip_eps_high` equal to `clip_eps`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GrpoOptions {
    /// How group rewards become advantages.
    pub advantage: AdvantageNorm,
    /// How token terms are normalised across the rows of a step.
    pub loss_agg: LossAgg,
    /// Upper edge of the clip band (`[1 - clip_eps, 1 + clip_eps_high]`).
    pub clip_eps_high: f32,
    /// Dynamic sampling: a step may draw up to this many times the groups it
    /// nominally needs (`ceil(grad_accum / group_size)`) while it holds fewer
    /// than `grad_accum` non-zero-advantage rows.
    pub resample_factor: usize,
    /// A queued row whose behaviour policy is more than this many optimiser
    /// steps older than the current policy is refused instead of trained.
    pub staleness_bound: u32,
    /// The distribution of the importance ratio; leave at [`RatioBasis::Trainer`].
    pub ratio_basis: RatioBasis,
    /// Compute and log the KL to the adapter-disabled reference even when
    /// `kl_beta` is zero (needs a model with LoRA tensors or `with_reference`).
    pub log_kl: bool,
}

impl Default for GrpoOptions {
    fn default() -> GrpoOptions {
        GrpoOptions { advantage: AdvantageNorm::default(), loss_agg: LossAgg::default(), clip_eps_high: 0.28, resample_factor: 3, staleness_bound: 1, ratio_basis: RatioBasis::Trainer, log_kl: true }
    }
}

/// GRPO's (and RFT/STaR's) hyperparameters. `seq_len` is the packed row's
/// length - MUST match the `t` the model was constructed with; `b` is `1`
/// (rollouts go through [`Model::logits_all`], a single-sequence oracle).
/// Drive a whole group's worth of rows through ONE optimiser step by setting
/// [`model::FitOpts::grad_accum`] to the number of rows a step should hold.
#[derive(Clone, Copy, Debug)]
pub struct GrpoConfig {
    /// Completions sampled per prompt. `1` selects the RFT/STaR degenerate
    /// path; `>= 2` is ordinary GRPO.
    pub group_size: usize,
    /// Lower clip-band half-width (`eps` in `clip(r, 1-eps, 1+eps_high)`).
    pub clip_eps: f32,
    /// k3 KL-to-reference coefficient. `0.0` puts no KL term in the loss.
    pub kl_beta: f32,
    /// Packed row length (`t`): prompt length + completion, at least.
    pub seq_len: usize,
    /// Sampling policy + stopping conditions for every rollout. Truncating
    /// the sampler (top-k, top-p) makes the behaviour policy differ from the
    /// trained one; such rows are flagged and counted
    /// ([`GrpoStats::truncated_sampling_rows`]).
    pub rollout: RolloutParams,
    /// `group_size == 1` (RFT/STaR) only: how many completions to sample per
    /// prompt before giving up on a verified-correct one.
    pub max_attempts: usize,
    /// See [`GrpoOptions`].
    pub options: GrpoOptions,
}

/// The frozen-reference logprob source the k3 KL term reads: `(task, sampled
/// completion) -> per-completion-token reference logprob`.
type RefLogprobFn = Box<dyn Fn(&Task, &Completion) -> Vec<f32>>;

/// One already-sampled, scored and packed training row.
struct PackedRow {
    tokens: Vec<u32>,
    targets: Vec<u32>,
    /// Behaviour log-probabilities, recomputed by the training forward.
    old_lp: Vec<f32>,
    /// Reference log-probabilities; meaningful only when `has_ref`.
    ref_lp: Vec<f32>,
    has_ref: bool,
    advantage: Vec<f32>,
    /// The completion span as verified and trained (no stop token).
    completion: Vec<u32>,
    /// Optimiser steps taken when this row's behaviour policy was current.
    version: u64,
    supervised: usize,
}

/// The sampling policy copy [`Grpo::with_rollout`] installs.
struct External {
    rollout: Box<dyn model::rollout::SyncRollout>,
    sync_every: usize,
    groups: usize,
}

/// One drawn group, before it is queued.
struct Drawn {
    rows: Vec<PackedRow>,
    /// The verified spans, in sample order, and their rewards.
    spans: Vec<Vec<u32>>,
    rewards: Vec<f32>,
    attempts: usize,
    zero_variance: bool,
    truncated: usize,
    eos_targets: usize,
    truncated_sampling_rows: usize,
    sampler_gap: f32,
}

/// One GRPO (or, with `cfg.group_size == 1`, RFT/STaR) training objective over
/// environment `E`'s tasks, scored by verifier `V`, for any `M: Model` that
/// provides [`training::TokenObjectives`].
pub struct Grpo<E: Environment, V: Verifier> {
    env: E,
    verifier: V,
    cfg: GrpoConfig,
    ref_logprobs: Option<RefLogprobFn>,
    pending: std::collections::VecDeque<PackedRow>,
    external: Option<External>,
    last_mean_reward: f32,
    last_kept_frac: f32,
    log: CycleLog,
    stats: GrpoStats,
    /// Optimiser steps taken so far (counted when [`Objective::step_verdict`] applies one).
    version: u64,
    verdict: Option<StepVerdict>,
    consecutive_nonfinite: u32,
}

impl<E: Environment, V: Verifier> Grpo<E, V> {
    pub fn new(env: E, verifier: V, cfg: GrpoConfig) -> Grpo<E, V> {
        assert!(cfg.group_size >= 1, "Grpo: group_size must be >= 1");
        Grpo {
            env,
            verifier,
            cfg,
            ref_logprobs: None,
            pending: std::collections::VecDeque::new(),
            external: None,
            last_mean_reward: 0.0,
            last_kept_frac: 0.0,
            log: CycleLog::default(),
            stats: GrpoStats::default(),
            version: 0,
            verdict: None,
            consecutive_nonfinite: 0,
        }
    }

    /// Attach the frozen-reference logprob source the k3 KL term reads. Called
    /// once per sampled completion, immediately after it is drawn. Without it
    /// the reference is the model with its adapters disabled.
    pub fn with_reference<F>(mut self, f: F) -> Grpo<E, V>
    where
        F: Fn(&Task, &Completion) -> Vec<f32> + 'static,
    {
        self.ref_logprobs = Some(Box::new(f));
        self
    }

    /// Sample through `rollout` (a serving engine holding its own copy of the
    /// policy, possibly on another device) instead of the trained model. The
    /// rollout is synced with the model's trainable tensors before the first
    /// group and then every `sync_every` groups. The trainer recomputes every
    /// sampled token's behaviour logprob on its own model when the group is
    /// packed: a different device or kernel never enters the probability ratio.
    pub fn with_rollout(mut self, rollout: Box<dyn model::rollout::SyncRollout>, sync_every: usize) -> Grpo<E, V> {
        assert!(sync_every >= 1, "Grpo: sync_every must be >= 1");
        self.external = Some(External { rollout, sync_every, groups: 0 });
        self
    }

    /// Attach an external [`CycleLog`] so a caller can observe what this
    /// objective sampled and trained on after it has been consumed.
    pub fn with_log(mut self, log: CycleLog) -> Grpo<E, V> {
        self.log = log;
        self
    }

    /// What the objective has done so far.
    pub fn stats(&self) -> &GrpoStats {
        &self.stats
    }

    /// The logit scale the trained distribution uses: `1/T` for a sampler at
    /// temperature `T`, `1` for greedy.
    fn logit_scale(&self) -> f32 {
        let sample = self.cfg.rollout.sample;
        if sample.is_greedy() || self.cfg.options.ratio_basis == RatioBasis::Sampler {
            1.0
        } else {
            1.0 / sample.temp
        }
    }

    fn truncating_sampler(&self) -> bool {
        let s = self.cfg.rollout.sample;
        !s.is_greedy() && (s.top_k > 0 || (s.top_p > 0.0 && s.top_p < 1.0))
    }

    fn band(&self) -> ClipBand {
        ClipBand { low: self.cfg.clip_eps, high: self.cfg.options.clip_eps_high }
    }

    /// Rollout parameters for the model's own rollout: its logits already
    /// carry the `1/T` scale, so it samples at temperature 1; a completion
    /// never exceeds the span the packed row can hold.
    fn own_params(&self, fit: usize) -> RolloutParams {
        let mut p = self.cfg.rollout;
        // The logits already carry the 1/T scale on the trainer basis.
        if !p.sample.is_greedy() && self.cfg.options.ratio_basis == RatioBasis::Trainer {
            p.sample.temp = 1.0;
        }
        p.max_new = p.max_new.min(fit);
        p
    }

    /// Draw and score one group (or one RFT attempt series) of `task`, and
    /// pack its informative rows. Reads `self` only: the loss probe uses it
    /// without touching the queue, the log or the counters.
    fn draw_group<M: Model>(&self, model: &M, task: &Task, roll: &mut dyn Rollout, own: bool, rng: &mut Rng) -> Drawn {
        let plen = task.prompt.len();
        assert!(plen > 0 && plen < self.cfg.seq_len, "Grpo: a prompt of {plen} tokens leaves no room in rows of {}", self.cfg.seq_len);
        let fit = self.cfg.seq_len - plen;
        let params = if own { self.own_params(fit) } else { self.cfg.rollout };
        let eos = self.cfg.rollout.eos;

        // Sample, cut every completion to the span the row can hold, and score that span.
        let mut kept: Vec<Completion> = Vec::new();
        let mut spans: Vec<Vec<u32>> = Vec::new();
        let mut rewards: Vec<f32> = Vec::new();
        let mut truncated = 0usize;
        let mut attempts = 0usize;
        let mut score = |c: Completion| {
            let cut = c.tokens.len() > fit;
            let c = if cut { Completion { tokens: c.tokens[..fit].to_vec(), logprobs: c.logprobs[..fit].to_vec(), stop: StopReason::MaxNew, eos_logprob: None } } else { c };
            let reward = self.verifier.verify(task, &[], &c.tokens).value;
            spans.push(c.tokens.clone());
            rewards.push(reward);
            kept.push(c);
            truncated += cut as usize;
            reward
        };
        let star = self.cfg.group_size == 1;
        if star {
            // RFT/STaR: bounded rejection sampling for one verified-correct completion.
            for _ in 0..self.cfg.max_attempts.max(1) {
                attempts += 1;
                let c = roll.sample_n(&task.prompt, 1, &params, rng).pop().expect("sample_n(1) returns exactly one completion");
                if score(c) > 0.0 {
                    break;
                }
            }
        } else {
            attempts = 1;
            for c in roll.sample_n(&task.prompt, self.cfg.group_size, &params, rng) {
                score(c);
            }
        }
        let advs: Vec<f32> = if star {
            let mut advs = vec![0.0; kept.len()];
            if rewards.last().is_some_and(|&r| r > 0.0) {
                *advs.last_mut().expect("one attempt was made") = 1.0;
            }
            advs
        } else {
            group_advantages_with(self.cfg.options.advantage, &rewards)
        };
        let zero_variance = advs.iter().all(|&a| a == 0.0);

        // Pack the informative rows; the stop token is a target when it fits.
        let mut rows = Vec::new();
        let mut eos_targets = 0usize;
        let mut sampler_lps: Vec<Vec<f32>> = Vec::new();
        for (c, &a) in kept.iter().zip(&advs).filter(|(_, &a)| a != 0.0) {
            let stop_target = eos.filter(|_| c.stop == StopReason::Eos && c.tokens.len() < fit);
            let mut trained = c.tokens.clone();
            let mut lps = c.logprobs.clone();
            if let Some(e) = stop_target {
                trained.push(e);
                lps.push(c.eos_logprob.unwrap_or(0.0));
                eos_targets += 1;
            }
            if trained.is_empty() {
                continue;
            }
            rows.push(self.pack(&task.prompt, c, &trained, &lps, a));
            sampler_lps.push(lps);
        }
        // The behaviour policy is the training forward's: record how far the sampler was from it.
        let truncated_sampling = own && self.truncating_sampler();
        let mut sampler_gap = 0.0f32;
        for (row, sampler) in rows.iter_mut().zip(&sampler_lps) {
            let trainer = trainer_logprobs(model, row);
            let positions: Vec<usize> = (0..row.targets.len()).filter(|&i| row.targets[i] != IGNORE).collect();
            for (&i, &s) in positions.iter().zip(sampler) {
                if own && !truncated_sampling && !self.cfg.rollout.sample.is_greedy() {
                    sampler_gap = sampler_gap.max((trainer[i] - s).abs());
                }
            }
            if self.cfg.options.ratio_basis == RatioBasis::Trainer {
                row.old_lp = trainer;
            }
        }
        self.attach_references(model, task, &kept, &advs, &mut rows);
        let truncated_sampling_rows = if truncated_sampling { rows.len() } else { 0 };
        Drawn { rows, spans, rewards, attempts, zero_variance, truncated, eos_targets, truncated_sampling_rows, sampler_gap }
    }

    /// Pack one completion (`trained` = its span plus the stop token when that
    /// is a target) into a row, at the alignment [`Completion::logprobs`]
    /// uses: position `prompt.len() - 1 + i` predicts `trained[i]`.
    fn pack(&self, prompt: &[u32], c: &Completion, trained: &[u32], logprobs: &[f32], adv: f32) -> PackedRow {
        let n = self.cfg.seq_len;
        let mut row = PackedRow {
            tokens: vec![0; n],
            targets: vec![IGNORE; n],
            old_lp: vec![0.0; n],
            ref_lp: vec![0.0; n],
            has_ref: false,
            advantage: vec![0.0; n],
            completion: c.tokens.clone(),
            version: self.version,
            supervised: 0,
        };
        pack_row(&mut row.tokens, &mut row.targets, &mut row.old_lp, &mut row.advantage, n, prompt, trained, logprobs, adv);
        row.supervised = row.targets.iter().filter(|&&t| t != IGNORE).count();
        row
    }

    /// Fill the rows' reference log-probabilities: from the caller's source,
    /// else from the model with its adapters disabled. A no-op when no KL is
    /// trained or logged.
    fn attach_references<M: Model>(&self, model: &M, task: &Task, kept: &[Completion], advs: &[f32], rows: &mut [PackedRow]) {
        let wanted = self.cfg.kl_beta > 0.0 || self.cfg.options.log_kl;
        if !wanted || rows.is_empty() {
            return;
        }
        if let Some(f) = &self.ref_logprobs {
            let completions = kept.iter().zip(advs).filter(|(_, &a)| a != 0.0).map(|(c, _)| c);
            for (row, c) in rows.iter_mut().zip(completions) {
                let lps = f(task, c);
                let positions: Vec<usize> = (0..row.targets.len()).filter(|&i| row.targets[i] != IGNORE).collect();
                for (j, &i) in positions.iter().enumerate() {
                    // A stop-token target has no reference value from the caller: it adds no KL.
                    row.ref_lp[i] = lps.get(j).copied().unwrap_or(row.old_lp[i]);
                }
                row.has_ref = true;
            }
            return;
        }
        match adapter_off_logprobs(model, rows) {
            Some(all) => {
                for (row, lp) in rows.iter_mut().zip(all) {
                    row.ref_lp = lp;
                    row.has_ref = true;
                }
            }
            None => assert!(self.cfg.kl_beta <= 0.0, "Grpo: kl_beta > 0.0 requires with_reference() or a model with LoRA adapters to disable"),
        }
    }

    /// Sample one group from the real task stream and queue its rows.
    fn refill<M: Model>(&mut self, model: &M, rng: &mut Rng) {
        let task = self.env.tasks(rng.next_u64()).into_iter().next().expect("Grpo: Environment::tasks produced no task");
        let mut external = self.external.take();
        let drawn = match &mut external {
            Some(ext) => {
                if ext.groups % ext.sync_every == 0 {
                    let names = model.optimized_params().unwrap_or_else(|| model.param_names());
                    let trained: std::collections::HashMap<String, Vec<f32>> = names.into_iter().map(|n| (n.clone(), model.read_weight(&n))).collect();
                    ext.rollout.sync(&trained).unwrap_or_else(|e| panic!("Grpo: syncing the rollout failed: {e}"));
                }
                ext.groups += 1;
                self.draw_group(model, &task, &mut *ext.rollout, false, rng)
            }
            None => self.draw_group(model, &task, &mut ModelRollout::new(model), true, rng),
        };
        self.external = external;
        for s in &drawn.spans {
            self.log.push_sampled(s.clone());
        }
        for &r in &drawn.rewards {
            self.log.push_reward(r);
        }
        self.last_mean_reward = drawn.rewards.iter().sum::<f32>() / drawn.rewards.len().max(1) as f32;
        self.last_kept_frac = if drawn.zero_variance { 0.0 } else { 1.0 };
        self.stats.record_group(drawn.attempts, drawn.zero_variance, drawn.truncated, drawn.eos_targets, drawn.truncated_sampling_rows, drawn.sampler_gap);
        self.pending.extend(drawn.rows);
    }

    /// Refuse queued rows whose behaviour policy is too old.
    fn refuse_stale(&mut self) {
        let (version, bound) = (self.version, self.cfg.options.staleness_bound);
        let stats = &mut self.stats;
        self.pending.retain(|row| {
            let age = version - row.version;
            let fresh = age <= u64::from(bound);
            if !fresh {
                stats.refuse(RowRefusal::Stale { age, bound });
            }
            fresh
        });
    }

    /// Up to `k` rows for one optimiser step, drawing groups while short.
    fn gather_rows<M: Model>(&mut self, model: &M, rng: &mut Rng, k: usize) -> Vec<PackedRow> {
        self.refuse_stale();
        let nominal = k.div_ceil(self.cfg.group_size);
        let budget = nominal * self.cfg.options.resample_factor.max(1);
        let mut drawn = 0;
        while self.pending.len() < k && drawn < budget {
            self.refill(model, rng);
            drawn += 1;
        }
        self.stats.resamples += drawn.saturating_sub(nominal);
        let take = k.min(self.pending.len());
        self.pending.drain(..take).collect()
    }

    /// Train `rows` as one optimiser step of `k` slots; leaves the verdict.
    fn train_rows<M: Model>(&mut self, model: &M, rows: Vec<PackedRow>, k: usize) -> f32 {
        self.stats.record_step(rows.len(), k);
        if rows.is_empty() {
            self.consecutive_nonfinite = 0;
            self.verdict = Some(StepVerdict::Skip("no row with a non-zero advantage after resampling".into()));
            self.log.set_stats(&self.stats);
            return 0.0;
        }
        let total_tokens: usize = rows.iter().map(|r| r.supervised).sum();
        // Each micro-step's gradient is divided by this, then by the loop's k: the sum is a mean over all the step's tokens.
        let denominator = match self.cfg.options.loss_agg {
            LossAgg::TokenSum => Some(total_tokens as f32 / k as f32),
            LossAgg::SeqMean => None,
        };
        let mut loss = 0.0;
        let mut finite = true;
        for row in &rows {
            self.log.push_trained(row.completion.clone());
            let (l, ok) = self.train_row(model, row, denominator);
            loss += l;
            finite &= ok;
        }
        finite &= gradients_are_finite(model);
        self.verdict = Some(self.rule(finite));
        self.log.set_stats(&self.stats);
        loss
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

    /// One row: forward, token weights, backward. Returns the row's share of
    /// the loss and whether everything it computed was finite (a row with a
    /// non-finite log-probability is not differentiated).
    fn train_row<M: Model>(&mut self, model: &M, row: &PackedRow, denominator: Option<f32>) -> (f32, bool) {
        model.set_batch(Batch::Lm { tokens: &row.tokens, targets: &row.targets });
        let _ = model.forward();
        let objectives = training::objectives(model);
        if let Some(d) = denominator {
            objectives.set_loss_denominator(d);
        }
        let new_lp = objectives.batch_token_logprobs();
        let supervised: Vec<usize> = (0..row.tokens.len()).filter(|&i| row.targets[i] != IGNORE).collect();
        if supervised.iter().any(|&i| !new_lp[i].is_finite()) {
            return (0.0, false);
        }
        let band = self.band();
        let kl_beta = self.cfg.kl_beta;
        let mut weights = vec![0f32; row.tokens.len()];
        let (mut total, mut kl_sum, mut ratio_dev) = (0.0f32, 0.0f32, 0.0f32);
        for &i in &supervised {
            let ref_lp = row.has_ref.then_some(row.ref_lp[i]);
            let term = token_term_banded(row.old_lp[i], new_lp[i], row.advantage[i], band, ref_lp.filter(|_| kl_beta > 0.0), kl_beta);
            weights[i] = term.weight;
            total += term.loss;
            if let Some(r) = ref_lp {
                kl_sum += k3_kl(r, new_lp[i]);
            }
            if row.version == self.version {
                ratio_dev = ratio_dev.max(((new_lp[i] - row.old_lp[i]).exp() - 1.0).abs());
            }
        }
        let finite = total.is_finite() && weights.iter().all(|w| w.is_finite());
        if finite {
            objectives.set_loss_weights(&weights);
            model.backward();
        }
        self.stats.record_row(ratio_dev, row.has_ref.then_some((kl_sum, row.supervised)));
        (total / denominator.unwrap_or(row.supervised as f32), finite)
    }
}

/// Place a completion in a row: `tokens` holds the prompt then the completion,
/// `targets[plen - 1 + i]` is `trained[i]`, and `old_lp`/`advantage` are set at
/// those positions. A completion that overruns `seq_len` is cut (callers cut
/// before scoring, so nothing is trained that was not rewarded).
#[allow(clippy::too_many_arguments)]
fn pack_row(tokens: &mut [u32], targets: &mut [u32], old_lp: &mut [f32], advantage: &mut [f32], seq_len: usize, prompt: &[u32], trained: &[u32], logprobs: &[f32], adv: f32) {
    let plen = prompt.len();
    if plen == 0 {
        return;
    }
    for (i, &t) in prompt.iter().chain(trained.iter()).take(seq_len).enumerate() {
        tokens[i] = t;
    }
    for (i, &t) in trained.iter().enumerate() {
        let idx = plen + i;
        if idx >= seq_len {
            break;
        }
        targets[idx - 1] = t;
        old_lp[idx - 1] = logprobs.get(i).copied().unwrap_or(0.0);
        advantage[idx - 1] = adv;
    }
}

/// The trained model's per-position log-probabilities of `row`.
fn trainer_logprobs<M: Model>(model: &M, row: &PackedRow) -> Vec<f32> {
    model.set_batch(Batch::Lm { tokens: &row.tokens, targets: &row.targets });
    let _ = model.forward();
    training::objectives(model).batch_token_logprobs()
}

/// The rows' per-position log-probabilities with every LoRA adapter disabled
/// (its `B` factor zeroed, which makes the adapter an exact no-op), or `None`
/// for a model without adapters. The adapters are restored before returning.
fn adapter_off_logprobs<M: Model>(model: &M, rows: &[PackedRow]) -> Option<Vec<Vec<f32>>> {
    let adapters: Vec<String> = model.param_names().into_iter().filter(|n| n.ends_with(".lora_b")).collect();
    if adapters.is_empty() {
        return None;
    }
    let saved: Vec<Vec<f32>> = adapters.iter().map(|n| model.read_weight(n)).collect();
    for (n, w) in adapters.iter().zip(&saved) {
        model.write_weight(n, &vec![0.0; w.len()]);
    }
    let out = rows.iter().map(|r| trainer_logprobs(model, r)).collect();
    for (n, w) in adapters.iter().zip(&saved) {
        model.write_weight(n, w);
    }
    model.poll_wait();
    Some(out)
}

/// Whether the accumulated gradients of the trained tensors are finite. A model
/// that does not expose its trained set is not inspected (its loss already was).
fn gradients_are_finite<M: Model>(model: &M) -> bool {
    let Some(names) = model.optimized_params() else { return true };
    names.iter().all(|n| model.read_grad(n).iter().all(|g| g.is_finite()))
}

impl<M: Model, E: Environment, V: Verifier> Objective<M> for Grpo<E, V> {
    fn regime(&self) -> &'static str {
        "grpo"
    }

    fn prepare(&mut self, model: &mut M) -> Result<(), Unsupported> {
        let objectives = training::require_token_objectives(model)?;
        objectives.enable_weighted_loss();
        objectives.set_logit_scale(self.logit_scale());
        Ok(())
    }

    fn micro_step(&mut self, model: &M, rng: &mut Rng) -> f32 {
        let rows = self.gather_rows(model, rng, 1);
        self.train_rows(model, rows, 1)
    }

    fn micro_steps(&mut self, model: &M, rng: &mut Rng, k: u32) -> f32 {
        let k = k.max(1) as usize;
        let rows = self.gather_rows(model, rng, k);
        self.train_rows(model, rows, k)
    }

    fn step_verdict(&mut self) -> StepVerdict {
        let verdict = self.verdict.take().unwrap_or(StepVerdict::Apply);
        if matches!(verdict, StepVerdict::Apply) {
            self.version += 1;
            self.stats.steps_applied += 1;
        }
        verdict
    }

    /// The loss of a freshly drawn group, forward only. The queue, the log and
    /// the counters are untouched: a probe is not training.
    fn loss_probe(&mut self, model: &M, rng: &mut Rng) -> f32 {
        let task = self.env.tasks(rng.next_u64()).into_iter().next().expect("Grpo: Environment::tasks produced no task");
        let drawn = self.draw_group(model, &task, &mut ModelRollout::new(model), true, rng);
        let (mut total, mut count) = (0.0f32, 0usize);
        for row in &drawn.rows {
            let lp = trainer_logprobs(model, row);
            for i in (0..row.tokens.len()).filter(|&i| row.targets[i] != IGNORE) {
                total += token_term_banded(row.old_lp[i], lp[i], row.advantage[i], self.band(), None, 0.0).loss;
                count += 1;
            }
        }
        if count > 0 {
            total / count as f32
        } else {
            0.0
        }
    }

    fn metrics(&self) -> Vec<(&'static str, f32)> {
        let mut m = vec![("grpo_mean_reward", self.last_mean_reward), ("grpo_kept_group_frac", self.last_kept_frac)];
        m.extend(self.stats.metrics());
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_row_aligns_targets_and_old_logprobs_to_the_completion_span() {
        let n = 8;
        let (mut tokens, mut targets, mut old_lp, mut adv) = (vec![0u32; n], vec![IGNORE; n], vec![0f32; n], vec![0f32; n]);
        pack_row(&mut tokens, &mut targets, &mut old_lp, &mut adv, n, &[1, 2, 3], &[4, 5, 6], &[-0.1, -0.2, -0.3], 1.5);
        assert_eq!(tokens, vec![1, 2, 3, 4, 5, 6, 0, 0]);
        // Position 2 (prompt.len()-1) predicts the first completion token.
        assert_eq!(targets, vec![IGNORE, IGNORE, 4, 5, 6, IGNORE, IGNORE, IGNORE]);
        assert_eq!(old_lp[2..5], [-0.1, -0.2, -0.3]);
        assert_eq!(adv[2..5], [1.5, 1.5, 1.5]);
        assert_eq!(adv[0], 0.0);
    }

    #[test]
    fn pack_row_cuts_a_completion_that_overruns_seq_len() {
        let n = 4;
        let (mut tokens, mut targets, mut old_lp, mut adv) = (vec![0u32; n], vec![IGNORE; n], vec![0f32; n], vec![0f32; n]);
        pack_row(&mut tokens, &mut targets, &mut old_lp, &mut adv, n, &[1, 2], &[3, 4, 5, 6], &[-0.1; 4], 1.0);
        assert_eq!(tokens, vec![1, 2, 3, 4]);
        assert_eq!(targets, vec![IGNORE, 3, 4, IGNORE]);
    }

    #[test]
    fn the_defaults_are_the_recommended_configuration() {
        let o = GrpoOptions::default();
        assert_eq!((o.advantage, o.loss_agg), (AdvantageNorm::MeanCentered, LossAgg::TokenSum));
        assert_eq!((o.clip_eps_high, o.resample_factor, o.staleness_bound), (0.28, 3, 1));
    }
}

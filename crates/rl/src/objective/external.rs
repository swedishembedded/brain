// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training language models from the rollouts an agent collected, for its
// clients. If your team needs expertise in reinforcement learning from externally collected agent trajectories then you
// can procure our services by sending an email to info@swedishembedded.com.

//! Policy-gradient training on rollouts collected outside brain.
//!
//! An orchestrator runs an agent, turns every model call into a chat record (the assistant turn of the call is the
//! action), scores each rollout with an independent verifier and writes the rollout's reward and its group (the
//! rollouts it is compared with) on each of the rollout's records ([`data::chat_rollout`]). [`ExternalPolicyGradient`]
//! turns the rewards of each group into the rollouts' advantages ([`PolicyGradientConfig::advantage_norm`],
//! [`super::advantage::group_advantages_with`]) and trains a LoRA adapter on the prepared dataset with the clipped
//! policy-gradient objective of GRPO/PPO on the supervised (action) tokens: records with a negative advantage push
//! their actions' probability down, those with a positive one pull it up. The per-token term
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
//!   log-probability of the adapter the run **started from**, measured by one forward pass over the example the first
//!   time a step draws it (a fresh adapter: the base), under a frozen copy of the start adapter so that it is the start
//!   policy's however far the adapter has moved by then. At the first optimiser step `r = 1` and the gradient is the plain
//!   on-policy estimator `-A * grad(new)`; the clip then limits how far later steps, and later epochs over the same
//!   rows, may move a token from where the run began. That is a proxy for the true behaviour policy, which is
//!   unknown, and is exact when the rollouts were sampled from the adapter the run starts from.
//! * **`ref`, the reference policy** ([`ReferencePolicy`]): the base model (the adapter disabled), or the adapter the
//!   run started from. They are the same for a fresh adapter. Measured by a forward pass over the example the first
//!   time a step draws it; the KL term is the k3 estimator `e^u - u - 1`, `u = ref - new`, at the sampled tokens. Its mean is
//!   reported even when `kl_beta` is zero ([`PolicyGradientConfig::log_kl`]).
//! * **Normalisation** ([`LossAgg`]): `SeqMean` divides each sequence's token terms by its own supervised-token
//!   count (GRPO as published: needs a batch of one sequence); `TokenSum` divides every token of an optimiser step by
//!   the step's mean count per micro-step, so all tokens weigh the same; `ConstantLength(n)` divides each sequence's
//!   sum by the constant `n` and averages over the sequences (Dr. GRPO), so neither a sequence's own length nor its
//!   step's token count rescales a token's gradient.
//! * **Advantages** are computed when the run opens, per group, from the rewards of the group's rollouts that reached
//!   an outcome, each rollout counted once however many records (calls) it made. A group of one has nothing to compare
//!   with and is refused; a truncated rollout takes no part, unless the run keeps truncated rollouts
//!   ([`PolicyGradientConfig::truncated`]), when it trains with the reward its collector gave it.
//! * **Records without a reward** are imitation examples: plain cross-entropy (weight 1, no KL), so a dataset may
//!   mix rollouts with supervised records.
//!
//! ## Self-distillation with privileged information
//!
//! A record may carry a teacher context instead of (or beside) an advantage ([`data::chat_rollout`]): the same
//! conversation with extra messages the student never saw. The teacher is the frozen model ([`PolicyGradientConfig::teacher`]:
//! the base, or the adapter the run started from) scoring the student's own sampled tokens after that context, and
//! every supervised token gets the dense advantage `A_t = log q_teacher(y_t | privileged context, y_<t) - old_t`, the
//! per-token reverse-KL signal (`old_t` is the token's behaviour log-probability). It is added to the record's scalar
//! advantage and trained through the one clipped step above; there is no second loss. Three guards keep it from
//! collapsing the policy: each `A_t` is clipped to `[-c, c]` ([`PolicyGradientConfig::advantage_bound`]), the sum of
//! `|A_t|` of a sequence may be capped ([`PolicyGradientConfig::sequence_advantage_cap`], so a few pivot tokens cannot
//! dominate a step: the sequence's advantages are scaled down together), and the k3 KL to the reference stays.
//! The teacher's rendering must fit the row and supervise exactly the student's tokens.
//!
//! ## On-policy distillation from a teacher model
//!
//! A record may instead carry another model's scores of the student's sampled answers
//! ([`super::onpolicy_distill`], [`data::chat_rollout`]). Per-token scores (`teacher_logprobs`) give every supervised
//! token the same dense advantage `A_t = log q_teacher(y_t | context) - old_t`, read from the record instead of scored
//! here, through the same guards; they are admitted only when the run's [`PolicyGradientConfig::teacher_tokenizer`]
//! says the teacher's tokenizer is the student's, and refused with a typed
//! [`super::onpolicy_distill::DistillRefusal`] otherwise. Per-answer scores (`teacher_logprob`) are the sequence-level
//! estimator, for any teacher: each rollout's `log q(text) - log p(text)` against the leave-one-out mean of the
//! rollouts of its group is added to its scalar advantage (after the reward advantage's floor). It is the
//! higher-variance path; a group's rollouts carry it all or none, at least two of them.
//!
//! The row of a record with images is read through the model's shared vision path (`Batch::LmExamples`), so images
//! train as they do in supervised fine-tuning: the tower is frozen and the loss covers the action tokens only.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

use data::binio;
use data::chat_rollout::TeacherStream;
pub use data::chat_rollout::TruncatedRollouts;
use data::loader::{BatchConfig, TokenDataset};
use data::rng::Rng;
use model::training::{self, Capability, Unsupported};
use model::{Batch, FitOpts, Model, ModelConfig, Objective, Split, StepVerdict, StepWork, IGNORE};

use super::advantage::{group_advantages_with, AdvantageFloor, AdvantageNorm};
use super::grpo::LossAgg;
use super::grpo_stats::{NonFiniteSteps, MAX_CONSECUTIVE_NONFINITE};
use super::grpo_term::ClipBand;
use super::onpolicy_distill::{sequence_advantages, token_advantage, DistillEstimator};
use super::numerics::{NumericsAccumulator, NumericsBounds, NumericsReport};
use super::pg_batch::{adapters_are_zero, adapters_can_be_disabled, backward_weighted_lp, weigh, with_adapters_disabled, AdapterState, BatchTerms, PgBatch, PgSpec};
use super::score_cache::{ExampleScores, ScoreCache, TokenAdvantages};

/// Consecutive optimiser steps the step guard may skip before the run is aborted.
pub const MAX_CONSECUTIVE_GUARD_SKIPS: u32 = 8;

/// Whether every gradient of the model is finite, and the sum of their squares (in f64), read back once. A model that
/// does not expose its gradients counts as finite with a zero norm.
fn gradient_scan<M: Model>(model: &M) -> (bool, f64) {
    let Some(names) = model.optimized_params() else { return (true, 0.0) };
    let (mut finite, mut sum_sq) = (true, 0.0f64);
    for name in names {
        for g in model.read_grad(&name) {
            finite &= g.is_finite();
            sum_sq += f64::from(g) * f64::from(g);
        }
    }
    (finite, sum_sq)
}

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

/// The correction for the gap between the engine that sampled a rollout and the trainer that updates on it.
///
/// With `None` the importance ratio is taken against the recorded behaviour log-probability, so every step-0 token
/// starts at `pi_trainer / pi_rollout`, not at one: the serve/train numeric gap eats the clip band before the first
/// update. With a correction the ratio is taken against the trainer's own start-of-run log-probability (one at step
/// 0, the clip band centred), and each token's policy term is scaled by the constant weight
/// `w = pi_trainer_start / pi_rollout` that corrects for the engine: truncated at `cap` ([`Self::Truncate`], TIS), or
/// dropped when `w` leaves `[1 / cap, cap]` ([`Self::Mask`], MIS, for a gap too large to correct by reweighting).
/// Costs one forward pass over each example the first time a step draws it (the start-of-run scoring); needs the
/// recorded behaviour log-probabilities ([`BehaviourSource::Recorded`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImportanceCorrection {
    #[default]
    None,
    Truncate { cap: f32 },
    Mask { cap: f32 },
}

impl ImportanceCorrection {
    /// The name the CLI and the adapter's provenance use.
    pub fn name(self) -> &'static str {
        match self {
            ImportanceCorrection::None => "none",
            ImportanceCorrection::Truncate { .. } => "truncate",
            ImportanceCorrection::Mask { .. } => "mask",
        }
    }

    /// Whether a correction is asked for.
    pub fn is_on(self) -> bool {
        self != ImportanceCorrection::None
    }

    /// The weight of a token whose trainer start-of-run log-probability is `trainer` and whose rollout engine
    /// recorded `rollout`.
    pub fn weight(self, trainer: f32, rollout: f32) -> f32 {
        let w = (trainer - rollout).exp();
        match self {
            ImportanceCorrection::None => 1.0,
            ImportanceCorrection::Truncate { cap } => w.min(cap),
            ImportanceCorrection::Mask { cap } if w > cap || w < 1.0 / cap => 0.0,
            ImportanceCorrection::Mask { .. } => w,
        }
    }
}

impl std::fmt::Display for ImportanceCorrection {
    /// The form [`std::str::FromStr`] reads: `none`, `truncate:<cap>` or `mask:<cap>`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImportanceCorrection::None => f.write_str("none"),
            ImportanceCorrection::Truncate { cap } => write!(f, "truncate:{cap}"),
            ImportanceCorrection::Mask { cap } => write!(f, "mask:{cap}"),
        }
    }
}

impl std::str::FromStr for ImportanceCorrection {
    type Err = String;
    /// `none`, `truncate:<cap>` or `mask:<cap>`.
    fn from_str(s: &str) -> Result<ImportanceCorrection, String> {
        let cap = |c: &str| c.parse::<f32>().map_err(|_| format!("the cap {c:?} is not a number"));
        match s.split_once(':') {
            None if s == "none" => Ok(ImportanceCorrection::None),
            Some(("truncate", c)) => Ok(ImportanceCorrection::Truncate { cap: cap(c)? }),
            Some(("mask", c)) => Ok(ImportanceCorrection::Mask { cap: cap(c)? }),
            _ => Err(format!("unknown importance correction {s:?}; expected \"none\", \"truncate:CAP\" or \"mask:CAP\"")),
        }
    }
}

/// Where the behaviour log-probability `old` of a token comes from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BehaviourSource {
    /// The sampler's own, recorded with the rollout. Every supervised token of every record that trains on the
    /// policy gradient must have one; the ratio is then the true importance ratio against the policy that sampled.
    #[default]
    Recorded,
    /// Where none was recorded, the log-probability under the policy the run started from. Exact only when the
    /// rollouts were sampled from that policy; otherwise a proxy, and the result says so.
    StartOfRunProxy,
}

impl BehaviourSource {
    /// The name the CLI, the served action and the adapter's provenance use.
    pub fn name(self) -> &'static str {
        match self {
            BehaviourSource::Recorded => "recorded",
            BehaviourSource::StartOfRunProxy => "start_of_run_proxy",
        }
    }
}

impl std::str::FromStr for BehaviourSource {
    type Err = String;
    fn from_str(s: &str) -> Result<BehaviourSource, String> {
        match s {
            "recorded" => Ok(BehaviourSource::Recorded),
            "start_of_run_proxy" => Ok(BehaviourSource::StartOfRunProxy),
            other => Err(format!("unknown behaviour source {other:?}; expected \"recorded\" or \"start_of_run_proxy\"")),
        }
    }
}

/// The scalar advantage of every example of a split, by `norm` over the rewards of its group: NaN for an imitation
/// example, 0 for a rollout without a reward of its own (a teacher context alone). Each rollout of a group counts
/// once (an example is one call, or one answer, of its rollout); a truncated rollout takes no part and trains on
/// nothing unless the run keeps it (`truncated`). A group that holds one rollout that takes part is refused, by the
/// contract's rule (`group_of_one`), as are two examples of one rollout that disagree (`rollout_mismatch`).
fn advantages(split: &data::chat_rollout::SplitRollouts, norm: AdvantageNorm, floor: AdvantageFloor, truncated: TruncatedRollouts) -> Result<Vec<f32>, String> {
    use data::chat_rollout::{group_members, GroupEntry, GroupFault, RecordRule};
    let graded: Vec<(usize, data::chat_rollout::Graded)> = split.examples.iter().enumerate().filter_map(|(e, r)| r.and_then(|r| r.graded).map(|g| (e, g))).collect();
    // Baseline-only outcomes take part in their groups like any rollout; no example trains on them.
    let entries: Vec<GroupEntry<u32, u32>> = graded.iter().map(|&(e, g)| GroupEntry { group: g.group, member: g.member, reward: g.reward, truncated: split.truncated[e] }).chain(split.baselines.iter().map(|g| GroupEntry { group: g.group, member: g.member, reward: g.reward, truncated: false })).collect();
    let (groups, faults) = group_members(&entries, truncated);
    if let Some(fault) = faults.first() {
        return Err(match fault {
            GroupFault::Mismatch { entry, field } => match graded.get(*entry) {
                Some((example, _)) => format!("{} (field {field}): example {example} differs from an earlier example of its rollout", RecordRule::RolloutMismatch.name()),
                None => format!("{} (field {field}): a baseline-only record differs from an earlier record of its rollout", RecordRule::RolloutMismatch.name()),
            },
            GroupFault::OfOne { group } => format!("{} (field group): group {:?} holds one rollout that takes part in it (truncated {}); a group-relative advantage needs at least two", RecordRule::GroupOfOne.name(), split.groups[*group as usize], truncated.name()),
        });
    }
    let mut by_member: HashMap<u32, f64> = HashMap::new();
    for members in groups.values() {
        let rewards: Vec<f64> = members.iter().map(|&(_, r)| r).collect();
        by_member.extend(members.iter().map(|&(m, _)| m).zip(floor.apply(group_advantages_with(norm, &rewards))));
    }
    // The sequence-level distillation signal: each rollout's log-ratio against the others of its group.
    let ratio: HashMap<u32, f64> = graded.iter().filter_map(|(_, g)| g.teacher_log_ratio.map(|r| (g.member, r))).collect();
    let baseline_only: HashSet<u32> = split.baselines.iter().map(|g| g.member).collect();
    for (group, members) in &groups {
        let completions: Vec<u32> = members.iter().map(|&(m, _)| m).filter(|m| !baseline_only.contains(m)).collect();
        let scored: Vec<f64> = completions.iter().filter_map(|m| ratio.get(m).copied()).collect();
        if scored.is_empty() {
            continue;
        }
        let rule = RecordRule::TeacherScores.name();
        let name = &split.groups[*group as usize];
        if scored.len() != completions.len() {
            return Err(format!("{rule}: group {name:?} holds {} rollouts with a teacher's per-answer score (\"teacher_logprob\") and {} without; the sequence-level baseline compares the completions of one prompt, so they carry it all or none", scored.len(), completions.len() - scored.len()));
        }
        if scored.len() < 2 {
            return Err(format!("{rule}: group {name:?} holds one rollout with a teacher's per-answer score; the sequence-level estimator's baseline is the other completions of the prompt, so it needs at least two"));
        }
        for (m, a) in completions.iter().zip(sequence_advantages(&scored)) {
            *by_member.get_mut(m).expect("every completion of a group has its reward advantage") += a;
        }
    }
    if let Some(a) = by_member.values().find(|a| !(**a as f32).is_finite()) {
        return Err(format!("a group's rewards give the advantage {a}, which is not a finite f32; rescale the rewards"));
    }
    Ok(split
        .examples
        .iter()
        .map(|r| match r {
            None => f32::NAN,
            Some(r) => r.graded.and_then(|g| by_member.get(&g.member)).map_or(0.0, |&a| a as f32),
        })
        .collect())
}

/// What a run would leave unread: a step draws `grad_accum` batches of `batch_size` records, so a run of `steps`
/// steps reads `steps * grad_accum * batch_size` records however many the dataset holds. A shorter run trains on a
/// subset, which is easy to mistake for training on the whole dataset; `None` when every record is drawn at least once.
fn coverage_note(opts: &FitOpts, records: usize) -> Option<String> {
    let drawn = u64::from(opts.steps) * u64::from(opts.grad_accum.max(1)) * u64::from(opts.batch_size.max(1));
    (drawn < records as u64).then(|| format!("{} steps of {} batches of {} records draw {drawn} of the dataset's {records} records ({:.1}%); the rest are never read. Raise steps or grad_accum to use them", opts.steps, opts.grad_accum.max(1), opts.batch_size.max(1), 100.0 * drawn as f64 / records as f64))
}

/// The knobs of [`ExternalPolicyGradient`]. [`Default`] is the recommended configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicyGradientConfig {
    /// Half-width of the clip band `[1 - eps, 1 + eps]` of the importance ratio, in `(0, 1)`: its lower edge, and its
    /// upper edge too unless [`Self::clip_eps_high`] widens it.
    pub clip_eps: f32,
    /// The upper edge `1 + eps_high` of the band when it is wider than the lower one ("clip-higher"): a low-probability
    /// token that turns out well can then keep gaining mass instead of being clipped at `1 + clip_eps`, which
    /// counters the entropy collapse a symmetric band causes. `None`: symmetric. At least `clip_eps`.
    pub clip_eps_high: Option<f32>,
    /// Dual-clip PPO's bound `c` on a negative advantage's term (Ye et al. 2020): the surrogate is
    /// `max(min(r*A, clip(r)*A), c*A)` there, so a token whose ratio ran past `c` costs at most `c*|A|` and gives no
    /// gradient. A token the behaviour policy all but never drew (a re-rendered proxy weighed against the start of the
    /// run) otherwise reaches a step at its full ratio: 1.6e6 and a gradient norm of 1.4e5 on real agent rollouts.
    /// Each step reports the share of terms it bounded. `None`: the plain clipped surrogate. Above one.
    pub dual_clip: Option<f32>,
    /// The engine-versus-trainer correction ([`ImportanceCorrection`]).
    pub importance: ImportanceCorrection,
    /// Skip, and log, the optimiser step of a step whose mean KL to the reference (the step's `kl`) exceeds this: a rare step with a
    /// few extreme log-ratios (KL 0.50 among neighbours of 0.003) must not reach the weights. Needs a reference
    /// measurement ([`Self::log_kl`] or a KL weight). `None`: no bound.
    pub max_step_kl: Option<f32>,
    /// Run as the numerics gate ([`super::numerics`]): no optimiser step is taken, and the run measures the gap between
    /// the recorded behaviour log-probabilities and the trainer's own, the effective sample size and the gradient norms of
    /// the batches it draws, judged against these bounds (`NumericsBounds::default()` reports and enforces nothing).
    /// Needs recorded behaviour log-probabilities and no importance correction.
    pub numerics: Option<NumericsBounds>,
    /// Read the keys and values the trainer's attention uses through the encoding of the KV cache the rollouts were
    /// sampled with (`Some(Int8)` or `Some(Fp8E4m3)`), so the teacher-forced forward has the server's numerics: its
    /// quantisation of the cache is the larger part of the serve/train gap where the weights are not themselves
    /// the noise (knowledge #446). `None`: the trainer's own precision. A trace names the cache it was served with
    /// (`brain.kv_cache`); a model whose trainer cannot apply it refuses the run.
    pub kv_roundtrip: Option<model::kv_tier::KvElement>,
    /// Skip, and log, the optimiser step of a step whose global gradient norm (after averaging over the step's
    /// micro-batches, before the optimiser's clip) exceeds this. `None`: no bound.
    pub max_step_grad_norm: Option<f32>,
    /// Coefficient of the k3 KL to the reference policy; `0.0` puts no KL term in the loss.
    /// The term is k3 weighted by the importance ratio to the behaviour policy, so its value and its expected
    /// gradient are those of `KL(pi || ref)`, the divergence reported (`grpo_term::token_term_weighted`).
    pub kl_beta: f32,
    pub reference: ReferencePolicy,
    /// Where the behaviour log-probabilities come from: the sampler's recorded ones ([`BehaviourSource::Recorded`],
    /// the default, exact) or, where none were recorded, the start-of-run policy's (an explicit, recorded proxy).
    pub behaviour: BehaviourSource,
    /// How token terms are normalised ([`LossAgg::SeqMean`], as GRPO publishes it, by default).
    pub loss_agg: LossAgg,
    /// The temperature the collector sampled with; the policy trained is `softmax(logits / temperature)`.
    pub temperature: f32,
    /// Measure and report the KL to the reference even when `kl_beta` is zero. Cost: one forward pass over each
    /// example the first time a step draws it (none after; the per-token KL comes from the step's own forward), and a
    /// second one only when a run that continues an adapter is measured against the base.
    pub log_kl: bool,
    /// Which frozen policy scores the student's tokens in the privileged context (records with a teacher context).
    pub teacher: ReferencePolicy,
    /// Clip each token's teacher advantage to `[-bound, bound]`; `None` leaves it unbounded.
    pub advantage_bound: Option<f32>,
    /// Cap the sum of `|A_t|` over a sequence's tokens (after clipping) by scaling the sequence's advantages down;
    /// `None` leaves it uncapped.
    pub sequence_advantage_cap: Option<f32>,
    /// How each group's rewards become the advantages of its rollouts.
    pub advantage_norm: AdvantageNorm,
    /// What is done with those advantages once computed against the whole group ([`AdvantageFloor::Zero`]: a rollout
    /// under the baseline sets it but carries no policy gradient; the KL term, when on, still reaches its tokens).
    pub advantage_floor: AdvantageFloor,
    /// What a rollout marked `"truncated": true` does: [`TruncatedRollouts::Mask`] (the default) never trains and takes
    /// no part in its group (DAPO's overlong filtering); [`TruncatedRollouts::Keep`] trains with its reward counted.
    pub truncated: TruncatedRollouts,
    /// How the tokenizer of the teacher model whose scores the records carry compares with the student's
    /// ([`data::tokenizer_identity::TokenizerIdentity::check`]): records with per-token teacher scores
    /// (`teacher_logprobs`) are trained only when it says they are identical ([`super::onpolicy_distill`]). `None`: no
    /// teacher model was named, and such records are refused.
    pub teacher_tokenizer: Option<data::tokenizer_identity::TokenizerIdentity>,
}

impl Default for PolicyGradientConfig {
    fn default() -> PolicyGradientConfig {
        PolicyGradientConfig { clip_eps: 0.2, clip_eps_high: None, dual_clip: Some(DEFAULT_DUAL_CLIP), importance: ImportanceCorrection::None, max_step_kl: None, max_step_grad_norm: None, numerics: None, kv_roundtrip: None, kl_beta: 0.02, reference: ReferencePolicy::Base, behaviour: BehaviourSource::Recorded, loss_agg: LossAgg::SeqMean, temperature: 1.0, log_kl: true, teacher: ReferencePolicy::Base, advantage_bound: None, sequence_advantage_cap: None, advantage_norm: AdvantageNorm::MeanCentered, advantage_floor: AdvantageFloor::None, truncated: TruncatedRollouts::Mask, teacher_tokenizer: None }
    }
}

/// A named policy-gradient algorithm: the settings that define it, laid over [`PolicyGradientConfig::default`]. A run
/// that names one may still set any other field (or override one of these); what it trained with is the configuration,
/// which a run records whole.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyGradientAlgorithm {
    /// GRPO as published: z-scored group advantages, each sequence's token mean, a symmetric clip band, the KL term.
    Grpo,
    /// Dr. GRPO: mean-centred advantages (no division by the group's spread), every sequence's sum over one constant
    /// length (the run's longest row), no KL term.
    DrGrpo,
    /// DAPO: z-scored advantages, clip-higher (`[0.8, 1.28]`), every token of a step weighed alike, truncated rollouts
    /// masked, no KL term.
    Dapo,
    /// RLOO: each rollout against the mean of the others of its group, every token of a step weighed alike.
    Rloo,
}

impl PolicyGradientAlgorithm {
    /// Every algorithm, in the order the wire lists them.
    pub const ALL: [PolicyGradientAlgorithm; 4] = [PolicyGradientAlgorithm::Grpo, PolicyGradientAlgorithm::DrGrpo, PolicyGradientAlgorithm::Dapo, PolicyGradientAlgorithm::Rloo];

    /// The algorithm's name on the wire (`algorithm`).
    pub fn name(self) -> &'static str {
        match self {
            PolicyGradientAlgorithm::Grpo => "grpo",
            PolicyGradientAlgorithm::DrGrpo => "dr-grpo",
            PolicyGradientAlgorithm::Dapo => "dapo",
            PolicyGradientAlgorithm::Rloo => "rloo",
        }
    }

    /// The algorithm a wire name names.
    ///
    /// # Errors
    /// The name is none of [`Self::name`]'s; the error lists them.
    pub fn parse(name: &str) -> Result<PolicyGradientAlgorithm, String> {
        PolicyGradientAlgorithm::ALL.into_iter().find(|a| a.name() == name).ok_or_else(|| format!("algorithm {name:?} is none of {}", PolicyGradientAlgorithm::ALL.map(PolicyGradientAlgorithm::name).join(", ")))
    }

    /// The configuration the algorithm names, for a run whose longest row is `max_block` tokens (Dr. GRPO's constant).
    pub fn config(self, max_block: u32) -> PolicyGradientConfig {
        let d = PolicyGradientConfig::default();
        match self {
            PolicyGradientAlgorithm::Grpo => PolicyGradientConfig { advantage_norm: AdvantageNorm::ZScore, loss_agg: LossAgg::SeqMean, clip_eps: 0.2, clip_eps_high: None, ..d },
            PolicyGradientAlgorithm::DrGrpo => PolicyGradientConfig { advantage_norm: AdvantageNorm::MeanCentered, loss_agg: LossAgg::ConstantLength(max_block), kl_beta: 0.0, ..d },
            PolicyGradientAlgorithm::Dapo => PolicyGradientConfig { advantage_norm: AdvantageNorm::ZScore, clip_eps: 0.2, clip_eps_high: Some(0.28), loss_agg: LossAgg::TokenSum, truncated: TruncatedRollouts::Mask, kl_beta: 0.0, ..d },
            PolicyGradientAlgorithm::Rloo => PolicyGradientConfig { advantage_norm: AdvantageNorm::LeaveOneOut, loss_agg: LossAgg::TokenSum, ..d },
        }
    }
}

impl model::training_plan::PlanCheck for PolicyGradientConfig {
    fn plan_name(&self) -> &'static str {
        "policy gradient"
    }

    fn check(&self, _opts: &FitOpts) -> Result<(), String> {
        self.validate()
    }
}

/// The dual clip [`PolicyGradientConfig`] defaults to: the `c = 3` of dual-clip PPO.
pub const DEFAULT_DUAL_CLIP: f32 = 3.0;

impl PolicyGradientConfig {
    /// Refuse a setting that has no meaning.
    pub fn validate(&self) -> Result<(), String> {
        if let Some(c) = self.dual_clip.filter(|c| !(c.is_finite() && *c > 1.0)) {
            return Err(format!("dual_clip is {c}; it must be a finite number above 1 (leave it unset for the plain clipped surrogate)"));
        }
        if !(self.clip_eps.is_finite() && self.clip_eps > 0.0 && self.clip_eps < 1.0) {
            return Err(format!("clip_eps is {}; it must lie in (0, 1)", self.clip_eps));
        }
        if let Some(high) = self.clip_eps_high.filter(|h| !(h.is_finite() && *h >= self.clip_eps && *h < 10.0)) {
            return Err(format!("clip_eps_high is {high}; it must be finite, at least clip_eps ({}) and below 10 (leave it unset for a symmetric band)", self.clip_eps));
        }
        if !(self.kl_beta.is_finite() && self.kl_beta >= 0.0) {
            return Err(format!("kl_beta is {}; it must be a finite number, at least 0", self.kl_beta));
        }
        if let ImportanceCorrection::Truncate { cap } | ImportanceCorrection::Mask { cap } = self.importance {
            if !(cap.is_finite() && cap > 1.0) {
                return Err(format!("the importance correction's cap is {cap}; it must be finite and above 1"));
            }
            if self.behaviour != BehaviourSource::Recorded {
                return Err("an importance correction weighs the trainer against the recorded behaviour log-probabilities; it needs behaviour \"recorded\"".to_string());
            }
        }
        if let Some(e) = self.kv_roundtrip {
            if !matches!(e, model::kv_tier::KvElement::Int8 | model::kv_tier::KvElement::Fp8E4m3) {
                return Err(format!("kv_roundtrip is {e:?}; only the lossy tiers (int8, fp8) change what the attention reads"));
            }
        }
        if let Some(bounds) = &self.numerics {
            bounds.validate()?;
            if self.behaviour != BehaviourSource::Recorded || self.importance.is_on() {
                return Err("the numerics gate compares the trainer with the recorded behaviour log-probabilities; it needs behaviour \"recorded\" and no importance correction".to_string());
            }
            if self.max_step_kl.is_some() || self.max_step_grad_norm.is_some() {
                return Err("the numerics gate takes no optimiser step, so a step guard has nothing to guard; set the gate's own bounds".to_string());
            }
        }
        for (name, value) in [("max_step_kl", self.max_step_kl), ("max_step_grad_norm", self.max_step_grad_norm)] {
            if let Some(v) = value.filter(|v| !(v.is_finite() && *v > 0.0)) {
                return Err(format!("{name} is {v}; it must be finite and positive (leave it unset for no guard)"));
            }
        }
        if self.max_step_kl.is_some() && !self.wants_reference() {
            return Err("max_step_kl bounds the KL to the reference, which is measured only with log_kl or a positive kl_beta".to_string());
        }
        self.loss_agg.validate()?;
        if !(self.temperature.is_finite() && self.temperature > 0.0) {
            return Err(format!("temperature is {}; it must be finite and positive", self.temperature));
        }
        for (name, value) in [("advantage_bound", self.advantage_bound), ("sequence_advantage_cap", self.sequence_advantage_cap)] {
            if let Some(v) = value.filter(|v| !(v.is_finite() && *v > 0.0)) {
                return Err(format!("{name} is {v}; it must be finite and positive (leave it unset for none)"));
            }
        }
        Ok(())
    }

    fn wants_reference(&self) -> bool {
        self.kl_beta > 0.0 || self.log_kl
    }
}

/// What one optimiser step measured. Losses are divided as the gradient is ([`LossAgg`]): means per supervised token,
/// or under [`LossAgg::ConstantLength`] each sequence's sum over the constant, averaged over the sequences.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
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
    /// [`Self::clip_fraction`] split by edge: the share of the signed tokens clipped at the upper edge (a positive
    /// advantage past `1 + eps_high`) and at the lower one (a negative advantage under `1 - eps`). They add up to it.
    pub clip_fraction_high: f32,
    pub clip_fraction_low: f32,
    /// Of the signed tokens, the share whose negative-advantage term the dual clip bounded (a ratio past `c`): no
    /// gradient, and not counted in [`Self::clip_fraction`]. Zero without a dual clip, and in records written before it.
    #[serde(default)]
    pub clip_fraction_dual: f32,
    /// The mean of `-log p(y)` over the policy tokens under the current policy: an estimator of the policy's entropy
    /// while the rollouts are on-policy (a falling series is the entropy collapse clip-higher counters).
    pub entropy_estimate: f32,
    /// Mean supervised tokens per sequence of the step: the response length.
    pub mean_response_tokens: f32,
    /// The global gradient norm of the step, averaged over its micro-batches, before the optimiser's clip.
    pub grad_norm: f32,
    /// The mean importance-sampling weight over the policy tokens and the share of them it dropped (one and zero
    /// without a correction).
    pub importance_weight_mean: f32,
    pub importance_masked_fraction: f32,
    /// Why the step's optimiser update was not taken (a guard bound was exceeded), when it was not.
    pub skipped: Option<String>,
    /// The KL to the reference, when one was measured: the mean over the policy tokens of the k3 estimator weighted by
    /// the importance ratio, an estimate of `KL(pi || ref)` that holds however far the policy has moved from the
    /// behaviour policy that drew the tokens (on-policy, the plain k3).
    pub kl: Option<f32>,
    /// The importance ratio over the policy tokens: mean, smallest and largest.
    pub ratio_mean: f32,
    pub ratio_min: f32,
    pub ratio_max: f32,
    /// Mean advantage of the step's sequences (imitation sequences excluded).
    pub mean_advantage: f32,
    /// Tokens trained as imitation (a record without an advantage).
    pub imitation_tokens: usize,
    /// Over the step's tokens that carry a teacher advantage (after the guards): their mean, the largest magnitude,
    /// and the fraction the bound clipped. All zero when the step has none.
    pub token_advantage_mean: f32,
    pub token_advantage_max: f32,
    pub token_advantage_clip_fraction: f32,
    /// Seconds the step spent scoring the examples it drew for the first time (zero when all had been scored).
    pub scoring_seconds: f32,
}

/// Everything a run measured, per optimiser step, and how it began.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
pub struct PolicyGradientStats {
    /// Where the behaviour log-probabilities of the run came from (see [`BehaviourSource`]); with
    /// `tokens_start_of_run_behavior` it says how much of the run's ratio is a proxy.
    pub behaviour: BehaviourSource,
    pub steps: Vec<PolicyGradientStep>,
    /// Optimiser steps not taken because a loss or gradient was not finite.
    pub steps_skipped_nonfinite: usize,
    /// Optimiser steps not taken because the step's KL or gradient norm was over its bound
    /// ([`PolicyGradientConfig::max_step_kl`], [`PolicyGradientConfig::max_step_grad_norm`]).
    pub steps_skipped_guard: usize,
    /// The numerics gate's measurements and verdict, for a run that was one ([`PolicyGradientConfig::numerics`]).
    pub numerics: Option<NumericsReport>,
    /// The serve-vs-train gap of every run, gate or not: each policy sequence's recorded behaviour log-probabilities
    /// against the trainer's under the start-of-run weights, over its tokens that have a recorded value, each sequence
    /// once (the first time a step draws it and the start-of-run value is known: from the start-of-run scoring, or from
    /// the step's own forward while no update has moved the weights). `None` until a sequence was measured. The
    /// records carry no routing, so this is the gap without routing replay.
    pub serve_train_gap: Option<NumericsReport>,
    /// Truncated rollouts of the dataset (`"truncated": true` on the record): never drawn, scored or trained on when
    /// the run masks them, trained like the others when it keeps them ([`PolicyGradientConfig::truncated`]).
    pub sequences_truncated: usize,
    /// Sequences of the dataset with an advantage, and without (imitation).
    pub sequences_with_advantage: usize,
    pub sequences_imitation: usize,
    /// Supervised tokens whose behaviour log-probability was recorded, and those that took the start-of-run value.
    pub tokens_recorded_behavior: usize,
    pub tokens_start_of_run_behavior: usize,
    /// Sequences with a teacher context, and of those (and of those with per-token teacher scores) the ones whose
    /// summed advantage the cap scaled down.
    pub sequences_with_teacher: usize,
    pub sequences_capped: usize,
    /// Sequences that carry a teacher model's per-token scores (token-level distillation) and a per-answer score
    /// (sequence-level distillation, [`super::onpolicy_distill`]).
    pub sequences_with_teacher_logprobs: usize,
    pub sequences_with_teacher_logprob: usize,
    /// Examples a step drew and the run therefore scored, by what was measured: the behaviour proxy under the start-of-run
    /// policy, the reference, and the teacher. An example no step draws is in none of them, and one drawn again is not
    /// scored again.
    pub scored_behavior: usize,
    pub scored_reference: usize,
    pub scored_teacher: usize,
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

    /// The mean share of the signed tokens the dual clip bounded, over all steps.
    pub fn mean_clip_fraction_dual(&self) -> Option<f32> {
        self.mean(|s| s.clip_fraction_dual)
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
    /// The examples a step may draw: every one whose rollout reached an outcome. A truncated rollout is masked as a
    /// whole ([`data::chat_rollout`]) by never being drawn, scored or counted among the training sequences.
    eligible: Vec<usize>,
    /// Per token of the stream; NaN where none was recorded.
    behavior: Vec<f32>,
    /// The teacher's renderings, when any example has one.
    teacher: Option<TeacherStream>,
    /// A teacher model's log-probability of each token of the stream (NaN where none was recorded), when any example
    /// carries them.
    teacher_logprobs: Option<Vec<f32>>,
}

/// The policy the run started from, which the behaviour proxy (and, when so configured, the reference and the teacher)
/// is measured under for as long as the run lasts.
enum StartPolicy {
    /// A fresh adapter: the policy is the base, exactly the adapters switched off.
    Base,
    /// A continued adapter: a frozen copy of its tensors, put back into the model while scoring once it has moved.
    Frozen(AdapterState),
}

/// What [`ExternalPolicyGradient::prepare`] fixed about the policies the examples are scored under.
struct Anchors {
    start: StartPolicy,
    wants_reference: bool,
}

impl Anchors {
    fn start_is_base(&self) -> bool {
        matches!(self.start, StartPolicy::Base)
    }
}

/// How many examples carry a teacher model's per-token scores (`lps`, per token of the stream, NaN where none),
/// refusing one that leaves a supervised token without a score: the record contract puts them on every answer, so a
/// gap is a dataset that was not prepared from such records.
fn check_teacher_logprobs(lps: &[f32], starts: &[usize], ends: &[usize], mask: &[bool]) -> Result<usize, String> {
    let mut scored = 0;
    for e in 0..starts.len() {
        if (starts[e]..ends[e]).all(|t| lps[t].is_nan()) {
            continue;
        }
        if let Some(t) = (starts[e]..ends[e]).find(|&t| mask[t] && lps[t].is_nan()) {
            return Err(format!("example {e}: supervised token {} carries no teacher log-probability and others of the example do; prepare the dataset again from its records", t - starts[e]));
        }
        scored += 1;
    }
    Ok(scored)
}

/// Refuse a teacher rendering that cannot be scored against its student: longer than a row, or supervising a different
/// number of tokens (a rendering that truncates, or a mask that disagrees). The tokens themselves were checked equal
/// when the dataset was prepared.
fn check_teacher(teacher: &TeacherStream, starts: &[usize], ends: &[usize], mask: &[bool], block_size: usize) -> Result<(), String> {
    for e in 0..starts.len() {
        let Some((ids, t_mask)) = teacher.example(e) else { continue };
        if ids.len() > block_size {
            return Err(format!("example {e}: the teacher's rendering is {} tokens, longer than the row of {block_size}; raise the block size or shorten the privileged context", ids.len()));
        }
        let student = (starts[e] + 1..ends[e]).filter(|&t| mask[t]).count();
        let seen = (1..t_mask.len()).filter(|&t| t_mask[t]).count();
        if student != seen {
            return Err(format!("example {e}: the teacher supervises {seen} tokens and the student {student}; the teacher's log-probabilities would not be those of the student's tokens"));
        }
    }
    Ok(())
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
    /// The importance-sampling weight of each position, when a correction is on.
    is_weight: Option<Vec<f32>>,
    has_ref: bool,
    /// The teacher advantages of the rows that have them, for the step's statistics.
    teacher: Vec<TokenAdvantages>,
    /// Per batch row, the example's log-probabilities under the start-of-run policy when the run scored them.
    start_lp: Vec<Option<Vec<f32>>>,
}

/// The step being accumulated.
#[derive(Default)]
struct StepAccumulator {
    terms: Vec<BatchTerms>,
    sequences: usize,
    advantage_sum: f64,
    advantaged_sequences: usize,
    teacher_tokens: usize,
    teacher_sum: f64,
    teacher_max: f32,
    teacher_clipped: usize,
}

/// Clipped policy-gradient training on a prepared dataset of externally collected rollouts. See the module docs.
pub struct ExternalPolicyGradient {
    cfg: PolicyGradientConfig,
    train: TokenDataset,
    batch_cfg: BatchConfig,
    rows: Rows,
    /// The student's supervision mask over the stream.
    mask: Vec<bool>,
    /// The dataset holds images: batches name their examples so the model reads them.
    tag_examples: bool,
    order: Option<Order>,
    /// What each drawn example was measured to be, filled the first time a step draws it.
    cache: ScoreCache,
    /// The largest token id the training split (and any teacher rendering) holds, which the model must have a row for.
    max_token: u32,
    /// Set by `prepare`.
    anchors: Option<Anchors>,
    /// The adapter may differ from the one the run started from: a step ran, or a resumed run restored its weights.
    weights_moved: bool,
    /// What the micro-steps since the last [`Objective::step_work`] measured.
    work: Option<StepWork>,
    scoring_seconds: f32,
    accumulating: StepAccumulator,
    stats: PolicyGradientStats,
    consecutive_nonfinite: u32,
    consecutive_guard: u32,
    gate: NumericsAccumulator,
    /// The serve-vs-train gap every run reports ([`PolicyGradientStats::serve_train_gap`]), and the examples measured.
    serve_gap: NumericsAccumulator,
    serve_gap_seen: HashSet<usize>,
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
        // The sidecars are the dataset's declaration that it holds rollouts. Without them every record would be
        // trained as an imitation example, which is a different objective than the one asked for.
        let Some(recorded) = data::chat_rollout::read(dir, "train", starts.len(), tokens.len())? else {
            let stray = ["train.blp.bin", "train.tch.u32.bin", "train.tch.mask.bin", "train.tch.ex.bin"].into_iter().find(|name| dir.join(name).exists());
            return Err(invalid(match stray {
                Some(name) => format!("{}: {name} is there but train.rollouts.json is not; a rollout dataset missing its rollout sidecar would train every record as imitation. Prepare the dataset again from its records", dir.display()),
                None => format!("{}: the training split carries no rollouts, so a policy gradient would train every record as imitation; use the supervised objective, or prepare the dataset from records with a \"reward\" and \"group\"", dir.display()),
            }));
        };
        let advantage = advantages(&recorded, cfg.advantage_norm, cfg.advantage_floor, cfg.truncated).map_err(|why| invalid(format!("{}: {why}", dir.display())))?;
        let (behavior, truncated) = (recorded.behavior, recorded.truncated);
        let eligible: Vec<usize> = (0..starts.len()).filter(|&e| cfg.truncated.counts(truncated[e])).collect();
        if let Some(t) = behavior.iter().position(|b| b.is_infinite() || *b > 1e-3) {
            return Err(invalid(format!("{}: token {t} has behaviour log-probability {}, which is not a log-probability", dir.display(), behavior[t])));
        }
        let teacher = data::chat_rollout::read_teacher(dir, "train", starts.len())?;
        if let Some(teacher) = &teacher {
            check_teacher(teacher, &starts, &ends, &mask, batch_cfg.block_size).map_err(invalid)?;
        }
        let teacher_logprobs = data::chat_rollout::read_teacher_logprobs(dir, "train", tokens.len())?;
        let scored_examples = match &teacher_logprobs {
            Some(lps) => check_teacher_logprobs(lps, &starts, &ends, &mask).map_err(|why| invalid(format!("{}: {why}", dir.display())))?,
            None => 0,
        };
        if teacher_logprobs.is_some() {
            if let Err(refusal) = DistillEstimator::TokenLevel.admit(cfg.teacher_tokenizer.as_ref()) {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, refusal));
            }
        }
        let mut stats = PolicyGradientStats {
            sequences_truncated: truncated.iter().filter(|t| **t).count(),
            sequences_with_teacher: teacher.as_ref().map_or(0, |t| (0..starts.len()).filter(|&e| t.example(e).is_some()).count()),
            sequences_with_teacher_logprobs: scored_examples,
            sequences_with_teacher_logprob: recorded.examples.iter().filter(|r| r.and_then(|r| r.graded).is_some_and(|g| g.teacher_log_ratio.is_some())).count(),
            ..Default::default()
        };
        for (e, &a) in advantage.iter().enumerate() {
            if !cfg.truncated.counts(truncated[e]) {
                continue;
            }
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
        if eligible.is_empty() {
            return Err(invalid(format!("{}: all {} records are truncated rollouts, which a policy gradient that masks them (truncated \"mask\") never trains on", dir.display(), starts.len())));
        }
        if stats.sequences_with_advantage == 0 {
            return Err(invalid(format!("{}: no training record carries a reward or a teacher context, so a policy gradient would train every record as imitation; use the supervised objective", dir.display())));
        }
        if cfg.behaviour == BehaviourSource::Recorded && stats.tokens_start_of_run_behavior > 0 {
            return Err(invalid(format!(
                "{}: {} of the {} supervised tokens of the records with an advantage have no recorded behaviour log-probability, so the ratio would not be taken against the policy that sampled them. Record them with the rollouts, or ask for the start-of-run proxy (behaviour \"start_of_run_proxy\"), which is exact only when the rollouts were sampled from the adapter this run starts from",
                dir.display(),
                stats.tokens_start_of_run_behavior,
                stats.tokens_start_of_run_behavior + stats.tokens_recorded_behavior
            )));
        }
        stats.behaviour = cfg.behaviour;
        if let Some(note) = coverage_note(opts, eligible.len()) {
            eprintln!("policy gradient: {note}");
        }
        let tag_examples = ["train", "val"].iter().any(|split| dir.join(format!("{split}.img.bin")).exists());
        let rows = Rows { starts, ends, advantage, eligible, behavior, teacher, teacher_logprobs };
        let cache = ScoreCache::create(dir, rows.starts.len())?;
        let max_token = tokens.iter().copied().chain(rows.teacher.iter().flat_map(|t| (0..rows.starts.len()).filter_map(|e| t.example(e)).flat_map(|(ids, _)| ids.iter().copied()))).max().unwrap_or(0);
        Ok(ExternalPolicyGradient { cfg, train, batch_cfg, rows, mask, tag_examples, order: None, cache, max_token, anchors: None, weights_moved: false, work: None, scoring_seconds: 0.0, accumulating: StepAccumulator::default(), stats, consecutive_nonfinite: 0, consecutive_guard: 0, gate: NumericsAccumulator::default(), serve_gap: NumericsAccumulator::default(), serve_gap_seen: HashSet::new(), verdict: None })
    }

    /// The scalar advantage example `example` of the training split trains with: its rollout's, computed from the
    /// rewards of its group when the run opened (0 for a rollout that trains on a teacher context alone); `None` for
    /// an imitation example or one past the split.
    pub fn advantage(&self, example: usize) -> Option<f32> {
        self.rows.advantage.get(example).copied().filter(|a| !a.is_nan())
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
        PgSpec { band: ClipBand { low: self.cfg.clip_eps, high: self.cfg.clip_eps_high.unwrap_or(self.cfg.clip_eps), dual: self.cfg.dual_clip }, kl_beta: self.cfg.kl_beta }
    }

    fn upload<M: Model>(&self, model: &M, tokens: &[u32], targets: &[u32], picks: &[Option<usize>]) {
        if self.tag_examples {
            model.set_batch(Batch::LmExamples { split: Split::Train, tokens, targets, examples: picks });
        } else {
            model.set_batch(Batch::Lm { tokens, targets });
        }
    }

    /// The tokens and targets of the batch whose rows are the examples `picks` names.
    fn token_batch(&self, picks: &[Option<usize>]) -> (Vec<u32>, Vec<u32>) {
        let (x, y) = self.train.example_batch(&self.batch_cfg, picks).expect("a prepared chat dataset indexes its examples");
        (x, y.iter().map(|&v| if v < 0 { IGNORE } else { v as u32 }).collect())
    }

    /// The batch whose rows are the examples `picks` names, with the per-position inputs of the objective. Every
    /// example must have been scored ([`Self::ensure_scored`]).
    fn lay_out(&self, picks: &[Option<usize>]) -> Laid {
        let (x, targets) = self.token_batch(picks);
        let (bs, bl) = (self.batch_cfg.batch_size, self.batch_cfg.block_size);
        let has_ref = self.anchors.as_ref().is_some_and(|a| a.wants_reference);
        let corrected = self.cfg.importance.is_on();
        let mut laid = Laid { tokens: x, targets, picks: picks.to_vec(), old_lp: vec![0.0; bs * bl], ref_lp: vec![0.0; bs * bl], advantage: vec![f32::NAN; bs * bl], is_weight: corrected.then(|| vec![1.0; bs * bl]), has_ref, teacher: Vec::new(), start_lp: vec![None; picks.len()] };
        for (b, pick) in picks.iter().enumerate() {
            let Some(e) = *pick else { continue };
            let scores = self.cache.get(e).unwrap_or_else(|err| panic!("policy-gradient score cache, example {e}: {err}"));
            for t in 0..bl.min(self.rows.ends[e] - self.rows.starts[e]) {
                let at = b * bl + t;
                if laid.targets[at] == IGNORE {
                    continue;
                }
                laid.advantage[at] = self.rows.advantage[e] + scores.teacher.as_ref().map_or(0.0, |a| a.value[t]);
                laid.old_lp[at] = self.old_logprob(e, t, scores.behavior.as_deref());
                if let (Some(weights), true) = (laid.is_weight.as_mut(), self.is_policy_example(e)) {
                    // The ratio is taken against the trainer's own start-of-run policy; the engine's gap to it is the weight.
                    let trainer = scores.behavior.as_ref().and_then(|lp| lp.get(t)).copied().expect("a corrected run scored the start-of-run log-probabilities of every policy token");
                    weights[at] = self.cfg.importance.weight(trainer, self.rows.behavior[self.rows.starts[e] + 1 + t]);
                    laid.old_lp[at] = trainer;
                }
                if has_ref {
                    laid.ref_lp[at] = scores.reference.as_ref().and_then(|r| r.get(t)).copied().unwrap_or(0.0);
                }
            }
            laid.teacher.extend(scores.teacher);
            laid.start_lp[b] = scores.behavior;
        }
        laid
    }

    /// The behaviour log-probability of the token row position `t` of example `e` predicts: the recorded one, else the
    /// start-of-run policy's (`start`, the example's measured log-probabilities under it).
    fn old_logprob(&self, e: usize, t: usize, start: Option<&[f32]>) -> f32 {
        let recorded = self.rows.behavior[self.rows.starts[e] + 1 + t];
        if recorded.is_nan() { start.and_then(|lp| lp.get(t)).copied().unwrap_or(0.0) } else { recorded }
    }

    /// Whether the example trains on the policy gradient, as opposed to imitating: it has an advantage of its own.
    fn is_policy_example(&self, e: usize) -> bool {
        !self.rows.advantage[e].is_nan()
    }

    /// Whether some supervised token of the example has no recorded behaviour log-probability, so the start-of-run
    /// policy stands in for it, and something reads it: the policy loss of an example with an advantage, or the
    /// teacher's advantage.
    fn needs_start_behavior(&self, e: usize) -> bool {
        let reads = self.is_policy_example(e) || self.rows.teacher.as_ref().is_some_and(|t| t.example(e).is_some()) || self.recorded_teacher(e).is_some();
        let corrected = self.cfg.importance.is_on() && self.is_policy_example(e);
        let mut supervised = ((self.rows.starts[e] + 1)..self.rows.ends[e]).filter(|&token| self.mask[token]);
        reads && supervised.any(|token| corrected || self.rows.behavior[token].is_nan())
    }

    /// The model's log-probability of every row position of the examples, by one forward pass per batch, in the
    /// policy the model is in now.
    fn score_policy<M: Model>(&self, model: &M, examples: &[usize]) -> Vec<Vec<f32>> {
        let (bs, bl) = (self.batch_cfg.batch_size, self.batch_cfg.block_size);
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(examples.len());
        for group in examples.chunks(bs) {
            let picks: Vec<Option<usize>> = (0..bs).map(|b| group.get(b).copied()).collect();
            let (tokens, targets) = self.token_batch(&picks);
            self.upload(model, &tokens, &targets, &picks);
            let _ = model.forward();
            let lp = training::objectives(model).batch_token_logprobs();
            for (b, &e) in group.iter().enumerate() {
                let len = (self.rows.ends[e] - self.rows.starts[e]).min(bl);
                out.push(lp[b * bl..b * bl + len].to_vec());
            }
        }
        out
    }

    /// The model's log-probability, in the teacher's context, of every supervised token of the examples (each has a
    /// teacher), by one forward pass per batch.
    fn score_teacher<M: Model>(&self, model: &M, teacher: &TeacherStream, examples: &[usize]) -> Vec<Vec<f32>> {
        let (bs, bl) = (self.batch_cfg.batch_size, self.batch_cfg.block_size);
        let mut out = Vec::with_capacity(examples.len());
        for group in examples.chunks(bs) {
            let (mut tokens, mut targets) = (vec![0u32; bs * bl], vec![IGNORE; bs * bl]);
            for (b, &e) in group.iter().enumerate() {
                let (ids, mask) = teacher.example(e).expect("filtered to examples with a teacher");
                tokens[b * bl..b * bl + ids.len()].copy_from_slice(ids);
                for t in 0..ids.len() - 1 {
                    if mask[t + 1] {
                        targets[b * bl + t] = ids[t + 1];
                    }
                }
            }
            model.set_batch(Batch::Lm { tokens: &tokens, targets: &targets });
            let _ = model.forward();
            let lp = training::objectives(model).batch_token_logprobs();
            for b in 0..group.len() {
                out.push((0..bl).filter(|&t| targets[b * bl + t] != IGNORE).map(|t| lp[b * bl + t]).collect());
            }
        }
        out
    }

    /// The teacher advantage of every supervised token of example `e` given the teacher's log-probabilities `lps` of
    /// them, guards applied; `start` is the example's start-of-run log-probabilities for tokens with no recorded ones.
    fn teacher_advantages(&self, e: usize, lps: &[f32], start: Option<&[f32]>, capped: &mut usize) -> TokenAdvantages {
        let bl = self.batch_cfg.block_size;
        let len = self.rows.ends[e] - self.rows.starts[e];
        let supervised = (0..bl.min(len)).filter(|&t| t + 1 < len && self.mask_at(e, t + 1));
        let (mut value, mut clipped) = (vec![f32::NAN; bl], vec![false; bl]);
        for (t, q) in supervised.zip(lps) {
            let raw = token_advantage(*q, self.old_logprob(e, t, start));
            value[t] = match self.cfg.advantage_bound {
                Some(c) if raw.abs() > c => {
                    clipped[t] = true;
                    raw.clamp(-c, c)
                }
                _ => raw,
            };
        }
        let magnitude: f32 = value.iter().filter(|v| !v.is_nan()).map(|v| v.abs()).sum();
        if let Some(cap) = self.cfg.sequence_advantage_cap.filter(|cap| magnitude > *cap) {
            value.iter_mut().filter(|v| !v.is_nan()).for_each(|v| *v *= cap / magnitude);
            *capped += 1;
        }
        TokenAdvantages { value, clipped }
    }

    /// A teacher model's recorded log-probability of every supervised token of example `e`, in row order, when the
    /// example carries them.
    fn recorded_teacher(&self, e: usize) -> Option<Vec<f32>> {
        let lps = self.rows.teacher_logprobs.as_ref()?;
        let (start, len) = (self.rows.starts[e], self.rows.ends[e] - self.rows.starts[e]);
        let supervised: Vec<f32> = (0..self.batch_cfg.block_size.min(len)).filter(|&t| t + 1 < len && self.mask_at(e, t + 1)).map(|t| lps[start + 1 + t]).collect();
        (!supervised.is_empty() && supervised.iter().all(|v| !v.is_nan())).then_some(supervised)
    }

    /// Whether the token at offset `i` of example `e` is supervised.
    fn mask_at(&self, e: usize, i: usize) -> bool {
        self.mask[self.rows.starts[e] + i]
    }

    /// Run `read` with the model in the policy the run started from.
    fn under_start<M: Model, R>(&self, model: &M, anchors: &Anchors, read: impl FnOnce() -> R) -> Result<R, Unsupported> {
        match &anchors.start {
            StartPolicy::Base => with_adapters_disabled(model, read).ok_or_else(|| Unsupported::of::<M>(Capability::ChatLoraFineTune)),
            // Until a step has run, or a resume restored weights, the live adapter is the start-of-run one.
            StartPolicy::Frozen(state) if self.weights_moved => Ok(state.with_applied(model, read)),
            StartPolicy::Frozen(_) => Ok(read()),
        }
    }

    /// Run `read` with the model's adapters disabled: the frozen base.
    fn under_base<M: Model, R>(model: &M, read: impl FnOnce() -> R) -> Result<R, Unsupported> {
        with_adapters_disabled(model, read).ok_or_else(|| Unsupported::of::<M>(Capability::ChatLoraFineTune))
    }

    /// Score the drawn examples that no earlier step drew, and keep what was measured. On-policy means the behaviour
    /// log-probabilities are the *start-of-run* policy's, whatever the adapter has become since: they are measured
    /// under the frozen start (a copy of the adapter tensors put in while scoring; the base itself for a fresh adapter,
    /// by switching the adapters off), never under the live adapter. The reference and the teacher are measured under
    /// the policies they name. Examples no step draws are never scored; one drawn again is read back, not scored again.
    fn ensure_scored<M: Model>(&mut self, model: &M, drawn: &[Option<usize>]) -> Result<(), Unsupported> {
        let mut seen = HashSet::new();
        let todo: Vec<usize> = drawn.iter().flatten().copied().filter(|&e| !self.cache.contains(e) && seen.insert(e)).collect();
        if todo.is_empty() {
            return Ok(());
        }
        let began = Instant::now();
        let anchors = self.anchors.take().expect("Objective::prepare fixes the anchors before the first step");
        let scored = self.score_examples(model, &anchors, &todo);
        self.anchors = Some(anchors);
        let (scores, capped) = scored?;
        for (&e, s) in todo.iter().zip(&scores) {
            self.stats.scored_behavior += usize::from(s.behavior.is_some());
            self.stats.scored_reference += usize::from(s.reference.is_some());
            self.stats.scored_teacher += usize::from(s.teacher.is_some());
            self.cache.put(e, s).unwrap_or_else(|err| panic!("policy-gradient score cache, example {e}: {err}"));
        }
        self.stats.sequences_capped += capped;
        self.scoring_seconds += began.elapsed().as_secs_f32();
        Ok(())
    }

    /// The scores of each of `todo`, and how many teacher sequences the cap scaled down.
    fn score_examples<M: Model>(&self, model: &M, anchors: &Anchors, todo: &[usize]) -> Result<(Vec<ExampleScores>, usize), Unsupported> {
        // The reference and the teacher are measured under the start when they are it (or the base is it).
        let reference_is_start = self.cfg.reference == ReferencePolicy::StartOfRun || anchors.start_is_base();
        let teacher_is_start = self.cfg.teacher == ReferencePolicy::StartOfRun || anchors.start_is_base();
        let wants_reference = |e: usize| anchors.wants_reference && self.is_policy_example(e);
        let under_start: Vec<usize> = todo.iter().copied().filter(|&e| self.needs_start_behavior(e) || (wants_reference(e) && reference_is_start)).collect();
        let under_base: Vec<usize> = todo.iter().copied().filter(|&e| wants_reference(e) && !reference_is_start).collect();
        let start_lp: HashMap<usize, Vec<f32>> = if under_start.is_empty() { HashMap::new() } else { under_start.iter().copied().zip(self.under_start(model, anchors, || self.score_policy(model, &under_start))?).collect() };
        let base_lp: HashMap<usize, Vec<f32>> = if under_base.is_empty() { HashMap::new() } else { under_base.iter().copied().zip(Self::under_base(model, || self.score_policy(model, &under_base))?).collect() };
        let mut teacher_lp: HashMap<usize, Vec<f32>> = HashMap::new();
        if let Some(teacher) = &self.rows.teacher {
            let with_teacher: Vec<usize> = todo.iter().copied().filter(|&e| teacher.example(e).is_some()).collect();
            if !with_teacher.is_empty() {
                let lps = if teacher_is_start { self.under_start(model, anchors, || self.score_teacher(model, teacher, &with_teacher))? } else { Self::under_base(model, || self.score_teacher(model, teacher, &with_teacher))? };
                teacher_lp = with_teacher.into_iter().zip(lps).collect();
            }
        }
        // A teacher model's scores were recorded with the rollout: read, not scored.
        teacher_lp.extend(todo.iter().filter_map(|&e| self.recorded_teacher(e).map(|lps| (e, lps))));
        let mut capped = 0;
        let scores = todo
            .iter()
            .map(|&e| {
                let behavior = self.needs_start_behavior(e).then(|| start_lp[&e].clone());
                let reference = wants_reference(e).then(|| if reference_is_start { start_lp[&e].clone() } else { base_lp[&e].clone() });
                let teacher = teacher_lp.get(&e).map(|lps| self.teacher_advantages(e, lps, behavior.as_deref(), &mut capped));
                ExampleScores { behavior, reference, teacher }
            })
            .collect();
        Ok((scores, capped))
    }

    /// What the run measured for an example a step has drawn; `None` for one no step has drawn yet.
    pub fn scores_of(&self, example: usize) -> Option<ExampleScores> {
        self.cache.contains(example).then(|| self.cache.get(example).expect("a kept score cache entry reads back"))
    }

    /// Bytes the scores hold in memory (a slot per example, however long the examples are) and on disk.
    pub fn score_cache_bytes(&self) -> (usize, u64) {
        (self.cache.resident_bytes(), self.cache.disk_bytes())
    }

    /// The step's statistics from what its micro-batches measured.
    fn close_step(&mut self, grad_norm: f32) {
        let acc = std::mem::take(&mut self.accumulating);
        let mut sum = BatchTerms::default();
        for t in &acc.terms {
            sum.loss += t.loss;
            sum.supervised += t.supervised;
            sum.policy_tokens += t.policy_tokens;
            sum.clipped += t.clipped;
            sum.clipped_high += t.clipped_high;
            sum.clipped_low += t.clipped_low;
            sum.dual_clipped += t.dual_clipped;
            sum.weight_masked += t.weight_masked;
            sum.weight_sum += t.weight_sum;
            sum.nll_sum += t.nll_sum;
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
            clip_fraction_high: if sum.signed_tokens > 0 { sum.clipped_high as f32 / sum.signed_tokens as f32 } else { 0.0 },
            clip_fraction_low: if sum.signed_tokens > 0 { sum.clipped_low as f32 / sum.signed_tokens as f32 } else { 0.0 },
            clip_fraction_dual: if sum.signed_tokens > 0 { sum.dual_clipped as f32 / sum.signed_tokens as f32 } else { 0.0 },
            entropy_estimate: if sum.policy_tokens > 0 { (sum.nll_sum / f64::from(policy)) as f32 } else { 0.0 },
            mean_response_tokens: if acc.sequences > 0 { sum.supervised as f32 / acc.sequences as f32 } else { 0.0 },
            grad_norm,
            importance_weight_mean: if sum.policy_tokens > 0 { (sum.weight_sum / f64::from(policy)) as f32 } else { 1.0 },
            importance_masked_fraction: if sum.policy_tokens > 0 { sum.weight_masked as f32 / policy } else { 0.0 },
            skipped: None,
            kl: (sum.kl_tokens > 0).then(|| sum.kl_sum / sum.kl_tokens as f32),
            ratio_mean: if sum.policy_tokens > 0 { (sum.ratio_sum / f64::from(policy)) as f32 } else { 1.0 },
            ratio_min: if sum.policy_tokens > 0 { sum.ratio_min } else { 1.0 },
            ratio_max: if sum.policy_tokens > 0 { sum.ratio_max } else { 1.0 },
            mean_advantage: if acc.advantaged_sequences > 0 { (acc.advantage_sum / acc.advantaged_sequences as f64) as f32 } else { 0.0 },
            imitation_tokens: sum.imitation_tokens,
            token_advantage_mean: if acc.teacher_tokens > 0 { (acc.teacher_sum / acc.teacher_tokens as f64) as f32 } else { 0.0 },
            token_advantage_max: acc.teacher_max,
            token_advantage_clip_fraction: if acc.teacher_tokens > 0 { acc.teacher_clipped as f32 / acc.teacher_tokens as f32 } else { 0.0 },
            scoring_seconds: std::mem::take(&mut self.scoring_seconds),
        });
    }

    /// Feed the numerics gate with the sequences of one micro-batch: for each policy sequence (a row with an advantage)
    /// the recorded behaviour log-probabilities against the trainer's, over its supervised tokens.
    fn measure_numerics(&mut self, laid: &Laid, new_lp: &[f32], band: ClipBand) {
        let bl = self.batch_cfg.block_size;
        for (b, pick) in laid.picks.iter().enumerate() {
            if pick.is_none_or(|e| !self.is_policy_example(e)) {
                continue;
            }
            let (mut old, mut new) = (Vec::new(), Vec::new());
            for t in (0..bl).filter(|&t| laid.targets[b * bl + t] != IGNORE) {
                old.push(laid.old_lp[b * bl + t]);
                new.push(new_lp[b * bl + t]);
            }
            self.gate.add_sequence(&old, &new, band);
        }
    }

    /// Feed the per-run serve-vs-train gap with the policy sequences of one micro-batch not measured before: the
    /// recorded behaviour log-probabilities against the start-of-run trainer's, over the tokens with a recorded value.
    /// The start-of-run value is the run's scoring of the example when it has one, else the forward `new_lp` while
    /// the weights are still the start-of-run ones; a sequence with neither waits for a later draw.
    fn measure_serve_gap(&mut self, laid: &Laid, new_lp: &[f32], band: ClipBand) {
        let bl = self.batch_cfg.block_size;
        for (b, pick) in laid.picks.iter().enumerate() {
            let Some(e) = *pick else { continue };
            if !self.is_policy_example(e) || self.serve_gap_seen.contains(&e) {
                continue;
            }
            let start = laid.start_lp[b].as_deref();
            if start.is_none() && self.weights_moved {
                continue;
            }
            let (mut old, mut new) = (Vec::new(), Vec::new());
            for t in (0..bl).filter(|&t| laid.targets[b * bl + t] != IGNORE) {
                let recorded = self.rows.behavior[self.rows.starts[e] + 1 + t];
                if recorded.is_nan() {
                    continue;
                }
                let trainer = start.map_or(new_lp[b * bl + t], |lp| lp[t]);
                // A non-finite value is the step's own finding (it skips the step); the gap is over measured tokens.
                if trainer.is_finite() {
                    old.push(recorded);
                    new.push(trainer);
                }
            }
            self.serve_gap.add_sequence(&old, &new, band);
            self.serve_gap_seen.insert(e);
        }
        self.stats.serve_train_gap = self.serve_gap.gap_report();
    }

    /// The verdict on the step just closed: a non-finite loss or gradient skips it (the run aborts after
    /// [`MAX_CONSECUTIVE_NONFINITE`] in a row), and so does a KL or gradient norm over the configured bound, logged
    /// with the values and counted (the run aborts after [`MAX_CONSECUTIVE_GUARD_SKIPS`] in a row: a bound that
    /// skips every step is wrong, or the policy has left the region it can be trained in).
    fn rule(&mut self, finite: bool) -> StepVerdict {
        if self.cfg.numerics.is_some() {
            // A measurement: the weights never move, and a batch that does not differentiate is a finding the report
            // carries (`grad_nonfinite`), not a reason to stop measuring.
            return StepVerdict::Skip("numerics gate: measuring, no optimiser step is taken".to_string());
        }
        if !finite {
            self.consecutive_guard = 0;
            self.consecutive_nonfinite += 1;
            self.stats.steps_skipped_nonfinite += 1;
            return if self.consecutive_nonfinite >= MAX_CONSECUTIVE_NONFINITE {
                StepVerdict::Abort(std::io::Error::new(std::io::ErrorKind::InvalidData, NonFiniteSteps { consecutive: self.consecutive_nonfinite }))
            } else {
                StepVerdict::Skip(format!("non-finite loss or gradient ({} in a row)", self.consecutive_nonfinite))
            };
        }
        self.consecutive_nonfinite = 0;
        let last = self.stats.steps.last_mut().expect("close_step ran");
        let over_kl = self.cfg.max_step_kl.zip(last.kl).filter(|(bound, kl)| kl > bound);
        let over_grad = self.cfg.max_step_grad_norm.filter(|bound| last.grad_norm > *bound);
        let why = match (over_kl, over_grad) {
            (Some((bound, kl)), _) => format!("the step's KL {kl:.4} is over the bound {bound} (gradient norm {:.3})", last.grad_norm),
            (None, Some(bound)) => format!("the step's gradient norm {:.3} is over the bound {bound}", last.grad_norm),
            (None, None) => {
                self.consecutive_guard = 0;
                return StepVerdict::Apply;
            }
        };
        last.skipped = Some(why.clone());
        self.stats.steps_skipped_guard += 1;
        self.consecutive_guard += 1;
        if self.consecutive_guard >= MAX_CONSECUTIVE_GUARD_SKIPS {
            return StepVerdict::Abort(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{MAX_CONSECUTIVE_GUARD_SKIPS} optimiser steps in a row were skipped by the step guard ({why}); the bound is too tight for this policy, or the policy has diverged")));
        }
        StepVerdict::Skip(why)
    }

    fn draw(&mut self) -> Vec<Option<usize>> {
        let order = self.order.as_mut().expect("Objective::begin positions the order before the first step");
        (0..self.batch_cfg.batch_size).map(|_| Some(self.rows.eligible[order.next()])).collect()
    }
}

impl<M: Model> Objective<M> for ExternalPolicyGradient {
    fn regime(&self) -> &'static str {
        "policy_gradient_external"
    }

    /// Opt the model into weighted-loss training, set the sampling temperature, and fix the policies the examples are
    /// scored under: the one the run starts from (the behaviour proxy; the reference and the teacher when they name it).
    /// Nothing is scored here: an example is scored when the first step that draws it asks ([`Self::ensure_scored`]).
    /// Runs before a resumed run restores its weights, so the start-of-run policy is the one captured now.
    fn prepare(&mut self, model: &mut M) -> Result<(), Unsupported> {
        let objectives = training::require_token_objectives(model)?;
        objectives.enable_weighted_loss();
        objectives.set_logit_scale(1.0 / self.cfg.temperature);
        if self.cfg.kv_roundtrip.is_some() {
            objectives.set_kv_roundtrip(self.cfg.kv_roundtrip).map_err(|_| Unsupported::of::<M>(Capability::KvRoundtrip))?;
        }
        let wants_reference = self.cfg.wants_reference() && self.stats.sequences_with_advantage > 0;
        let start = if adapters_are_zero(model) { StartPolicy::Base } else { StartPolicy::Frozen(AdapterState::capture(model)) };
        // A frozen base is read by switching adapters off, which needs adapters to switch off.
        let base_is_read = matches!(start, StartPolicy::Base) || (wants_reference && self.cfg.reference == ReferencePolicy::Base) || (self.stats.sequences_with_teacher > 0 && self.cfg.teacher == ReferencePolicy::Base);
        if base_is_read && !adapters_can_be_disabled(model) {
            return Err(Unsupported::of::<M>(Capability::ChatLoraFineTune));
        }
        self.anchors = Some(Anchors { start, wants_reference });
        Ok(())
    }

    fn check_data(&mut self, model: &M) -> Result<(), std::io::Error> {
        let vocab = model.config().vocab();
        if self.max_token >= vocab {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("the dataset holds token id {} but the model's vocabulary is {vocab}", self.max_token)));
        }
        Ok(())
    }

    fn begin(&mut self, seed: u64, draws_done: u64) {
        self.order = Some(Order::new(seed, self.rows.eligible.len(), draws_done));
    }

    /// The record order is the loop's ([`Self::begin`]); what a resume needs from the objective is the history it
    /// reports, so a resumed run's statistics are the whole run's and its step numbers continue.
    fn cursor(&self) -> Option<serde_json::Value> {
        let mut seen: Vec<usize> = self.serve_gap_seen.iter().copied().collect();
        seen.sort_unstable();
        Some(serde_json::json!({ "steps": self.stats.steps, "steps_skipped_nonfinite": self.stats.steps_skipped_nonfinite, "steps_skipped_guard": self.stats.steps_skipped_guard, "serve_gap": self.serve_gap, "serve_gap_seen": seen }))
    }

    fn restore_cursor(&mut self, cursor: &serde_json::Value) -> Result<(), std::io::Error> {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct History {
            steps: Vec<PolicyGradientStep>,
            steps_skipped_nonfinite: usize,
            steps_skipped_guard: usize,
            #[serde(default)]
            serve_gap: NumericsAccumulator,
            #[serde(default)]
            serve_gap_seen: Vec<usize>,
        }
        let history: History = serde_json::from_value(cursor.clone()).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("the policy-gradient history of the saved state: {e}")))?;
        if history.steps.iter().enumerate().any(|(i, s)| s.step != i + 1) {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "the policy-gradient history of the saved state does not number its steps 1, 2, ..."));
        }
        self.stats.steps = history.steps;
        self.stats.steps_skipped_nonfinite = history.steps_skipped_nonfinite;
        self.stats.steps_skipped_guard = history.steps_skipped_guard;
        if let Some(&e) = history.serve_gap_seen.iter().find(|&&e| e >= self.rows.advantage.len()) {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("the serve/train gap of the saved state measured example {e}, which this dataset does not hold")));
        }
        self.serve_gap = history.serve_gap;
        self.serve_gap_seen = history.serve_gap_seen.into_iter().collect();
        self.stats.serve_train_gap = self.serve_gap.gap_report();
        Ok(())
    }

    /// The weights differ from the start-of-run ones exactly when an update was applied (also true of a
    /// resumed run, which restored the weights it stopped with). A step the update gate skipped moved nothing.
    fn updates_applied(&mut self, applied: u32) {
        self.weights_moved = applied > 0;
    }

    fn micro_step(&mut self, model: &M, rng: &mut Rng) -> f32 {
        self.micro_steps(model, rng, 1)
    }

    /// `k` micro-batches of one optimiser step. Their gradients accumulate; a model with a loss denominator divides
    /// each by the step's mean supervised-token count under [`LossAgg::TokenSum`].
    fn micro_steps(&mut self, model: &M, _rng: &mut Rng, k: u32) -> f32 {
        // Preparation is the draws, the scoring of examples no earlier step drew and the layout; the passes are
        // measured around each forward and backward, as supervised training measures them.
        let clock = std::time::Instant::now();
        let drawn: Vec<Vec<Option<usize>>> = (0..k.max(1)).map(|_| self.draw()).collect();
        // `prepare` checked that the model can be scored under every policy the run names.
        self.ensure_scored(model, &drawn.concat()).expect("Objective::prepare checked the scoring policies");
        let laid: Vec<Laid> = drawn.iter().map(|picks| self.lay_out(picks)).collect();
        let supervised: usize = laid.iter().map(|l| l.targets.iter().filter(|&&t| t != IGNORE).count()).sum();
        let useful = laid.iter().flat_map(|l| l.picks.iter().flatten()).map(|&e| (self.rows.ends[e] - self.rows.starts[e]).min(self.batch_cfg.block_size) as u64).sum();
        let mut work = StepWork { useful_tokens: useful, supervised_positions: supervised as u64, prepare_secs: clock.elapsed().as_secs_f64(), ..StepWork::default() };
        let denominator = self.cfg.loss_agg.denominator(self.batch_cfg.batch_size, supervised, laid.len());
        let spec = self.spec();
        let mut loss = 0.0;
        let mut finite = true;
        for l in &laid {
            let pass = std::time::Instant::now();
            self.upload(model, &l.tokens, &l.targets, &l.picks);
            work.prepare_secs += pass.elapsed().as_secs_f64();
            let pass = std::time::Instant::now();
            let _ = model.forward();
            work.forward_secs += pass.elapsed().as_secs_f64();
            work.computed_tokens += model.computed_tokens().unwrap_or(l.tokens.len() as u64);
            let batch = PgBatch { targets: &l.targets, old_lp: &l.old_lp, ref_lp: l.has_ref.then_some(&l.ref_lp[..]), advantage: &l.advantage, is_weight: l.is_weight.as_deref() };
            let pass = std::time::Instant::now();
            let (terms, new_lp) = backward_weighted_lp(model, &batch, &spec, denominator);
            work.backward_secs += pass.elapsed().as_secs_f64();
            if self.cfg.numerics.is_some() {
                self.measure_numerics(l, &new_lp, spec.band);
            }
            self.measure_serve_gap(l, &new_lp, spec.band);
            finite &= terms.finite;
            loss += terms.normalised_loss(denominator);
            for e in l.picks.iter().flatten() {
                self.accumulating.sequences += 1;
                if !self.rows.advantage[*e].is_nan() {
                    self.accumulating.advantaged_sequences += 1;
                    self.accumulating.advantage_sum += f64::from(self.rows.advantage[*e]);
                }
            }
            for adv in &l.teacher {
                for (v, c) in adv.value.iter().zip(&adv.clipped).filter(|(v, _)| !v.is_nan()) {
                    self.accumulating.teacher_tokens += 1;
                    self.accumulating.teacher_sum += f64::from(*v);
                    self.accumulating.teacher_max = self.accumulating.teacher_max.max(v.abs());
                    self.accumulating.teacher_clipped += usize::from(*c);
                }
            }
            self.accumulating.terms.push(terms);
        }
        let (grads_finite, grad_sq) = gradient_scan(model);
        finite &= grads_finite;
        let grad_norm = (grad_sq.sqrt() / f64::from(k.max(1))) as f32;
        if let Some(bounds) = &self.cfg.numerics {
            self.gate.add_grad_norm((grads_finite && grad_norm.is_finite()).then_some(grad_norm));
            self.stats.numerics = Some(self.gate.report(bounds));
        }
        self.close_step(grad_norm);
        self.verdict = Some(self.rule(finite));
        self.work = Some(work);
        loss
    }

    fn step_work(&mut self) -> Option<StepWork> {
        self.work.take()
    }

    fn step_verdict(&mut self) -> StepVerdict {
        self.verdict.take().unwrap_or(StepVerdict::Apply)
    }

    /// The loss of freshly drawn examples on the forward alone: nothing is queued, counted or differentiated.
    fn loss_probe(&mut self, model: &M, rng: &mut Rng) -> f32 {
        let eligible = self.rows.eligible.len();
        let picks: Vec<Option<usize>> = (0..self.batch_cfg.batch_size).map(|_| Some(self.rows.eligible[rng.gen_range_inclusive(0, eligible as i64 - 1) as usize])).collect();
        self.ensure_scored(model, &picks).expect("Objective::prepare checked the scoring policies");
        let laid = self.lay_out(&picks);
        self.upload(model, &laid.tokens, &laid.targets, &laid.picks);
        let _ = model.forward();
        let batch = PgBatch { targets: &laid.targets, old_lp: &laid.old_lp, ref_lp: laid.has_ref.then_some(&laid.ref_lp[..]), advantage: &laid.advantage, is_weight: laid.is_weight.as_deref() };
        let (_, terms) = weigh(&training::objectives(model).batch_token_logprobs(), &batch, &self.spec());
        terms.normalised_loss(self.cfg.loss_agg.denominator(self.batch_cfg.batch_size, terms.supervised, 1))
    }

    fn step_log(&self) -> Option<String> {
        let last = self.stats.steps.last()?;
        let kl = last.kl.map(|kl| format!("kl {kl:.4}  ")).unwrap_or_default();
        let teacher = if self.stats.sequences_with_teacher + self.stats.sequences_with_teacher_logprobs > 0 { format!("  adv mean {:+.3} max {:.3} clipped {:.3}", last.token_advantage_mean, last.token_advantage_max, last.token_advantage_clip_fraction) } else { String::new() };
        let scoring = if last.scoring_seconds > 0.0 { format!("  scoring {:.1}s", last.scoring_seconds) } else { String::new() };
        let skipped = if last.skipped.is_some() { "  SKIPPED" } else { "" };
        Some(format!("{kl}clip {:.3} (hi {:.3} lo {:.3} dual {:.3})  ratio {:.3}  entropy~{:.3}  len {:.0}  gnorm {:.3}{teacher}{scoring}{skipped}", last.clip_fraction, last.clip_fraction_high, last.clip_fraction_low, last.clip_fraction_dual, last.ratio_mean, last.entropy_estimate, last.mean_response_tokens, last.grad_norm))
    }

    fn metrics(&self) -> Vec<(&'static str, f32)> {
        let Some(last) = self.stats.steps.last() else { return Vec::new() };
        let mut m = vec![("pg_loss", last.loss), ("pg_clip_fraction", last.clip_fraction), ("pg_clip_fraction_high", last.clip_fraction_high), ("pg_clip_fraction_low", last.clip_fraction_low), ("pg_clip_fraction_dual", last.clip_fraction_dual), ("pg_entropy_estimate", last.entropy_estimate), ("pg_response_tokens", last.mean_response_tokens), ("pg_grad_norm", last.grad_norm), ("pg_importance_weight_mean", last.importance_weight_mean), ("pg_importance_masked_fraction", last.importance_masked_fraction), ("pg_ratio_mean", last.ratio_mean), ("pg_ratio_min", last.ratio_min), ("pg_ratio_max", last.ratio_max), ("pg_mean_advantage", last.mean_advantage), ("pg_scoring_seconds", last.scoring_seconds)];
        if let Some(kl) = last.kl {
            m.push(("pg_kl", kl));
        }
        if self.stats.sequences_with_teacher + self.stats.sequences_with_teacher_logprobs > 0 {
            m.extend([("pg_token_advantage_mean", last.token_advantage_mean), ("pg_token_advantage_max", last.token_advantage_max), ("pg_token_advantage_clip_fraction", last.token_advantage_clip_fraction)]);
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each setting's written form reads back as the same setting: what a served contract declares as a default is
    /// a value the request accepts.
    #[test]
    fn the_written_settings_read_back() {
        for importance in [ImportanceCorrection::None, ImportanceCorrection::Truncate { cap: 2.0 }, ImportanceCorrection::Mask { cap: 1.5 }] {
            assert_eq!(importance.to_string().parse::<ImportanceCorrection>(), Ok(importance));
        }
        for agg in [LossAgg::SeqMean, LossAgg::TokenSum, LossAgg::ConstantLength(4096)] {
            assert_eq!(agg.to_string().parse::<LossAgg>(), Ok(agg));
        }
        for bad in ["constant-length", "constant-length:0", "constant-length:-3", "mean"] {
            assert!(bad.parse::<LossAgg>().is_err(), "{bad}");
        }
        for reference in [ReferencePolicy::Base, ReferencePolicy::StartOfRun] {
            assert_eq!(reference.name().parse::<ReferencePolicy>(), Ok(reference));
        }
    }

    /// Each algorithm sets what defines it and leaves every other setting the default's; its wire name reads back.
    #[test]
    fn each_algorithm_expands_to_the_settings_that_define_it() {
        let d = PolicyGradientConfig::default();
        for algorithm in PolicyGradientAlgorithm::ALL {
            assert_eq!(PolicyGradientAlgorithm::parse(algorithm.name()), Ok(algorithm));
            let c = algorithm.config(1024);
            assert!(c.validate().is_ok(), "{algorithm:?}");
            // What no algorithm decides stays the default's.
            assert_eq!((c.importance, c.reference, c.behaviour, c.temperature, c.teacher, c.max_step_kl), (d.importance, d.reference, d.behaviour, d.temperature, d.teacher, d.max_step_kl), "{algorithm:?}");
        }
        let grpo = PolicyGradientAlgorithm::Grpo.config(1024);
        assert_eq!((grpo.advantage_norm, grpo.loss_agg, grpo.clip_eps, grpo.clip_eps_high, grpo.kl_beta), (AdvantageNorm::ZScore, LossAgg::SeqMean, 0.2, None, d.kl_beta));
        let dr = PolicyGradientAlgorithm::DrGrpo.config(1024);
        assert_eq!((dr.advantage_norm, dr.loss_agg, dr.kl_beta), (AdvantageNorm::MeanCentered, LossAgg::ConstantLength(1024), 0.0));
        let dapo = PolicyGradientAlgorithm::Dapo.config(1024);
        assert_eq!((dapo.advantage_norm, dapo.clip_eps, dapo.clip_eps_high, dapo.loss_agg, dapo.truncated, dapo.kl_beta), (AdvantageNorm::ZScore, 0.2, Some(0.28), LossAgg::TokenSum, TruncatedRollouts::Mask, 0.0));
        let rloo = PolicyGradientAlgorithm::Rloo.config(1024);
        assert_eq!((rloo.advantage_norm, rloo.loss_agg, rloo.kl_beta), (AdvantageNorm::LeaveOneOut, LossAgg::TokenSum, d.kl_beta));
        assert!(PolicyGradientAlgorithm::parse("ppo").unwrap_err().contains("grpo, dr-grpo, dapo, rloo"));
    }

    /// A run that draws fewer records than the dataset holds says how many it leaves unread; one that covers the
    /// dataset says nothing.
    #[test]
    fn a_run_that_leaves_records_unread_says_so() {
        let run = |steps, grad_accum, batch_size| FitOpts { steps, grad_accum, batch_size, ..FitOpts::default() };
        let note = coverage_note(&run(100, 1, 1), 2400).expect("100 of 2400 records");
        assert!(note.contains("100 of the dataset's 2400") && note.contains("4.2%"), "{note}");
        assert!(coverage_note(&run(100, 24, 1), 2400).is_none(), "2400 draws read every record");
        assert!(coverage_note(&run(10, 4, 8), 300).is_none(), "320 draws");
        assert!(coverage_note(&run(10, 4, 8), 321).is_some());
    }
}

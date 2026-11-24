// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for stable policy-gradient training of language models, online and from
// recorded rollouts alike, for its clients. If your team needs expertise in clipped policy-gradient objectives that
// keep a rare extreme step from reaching the weights then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The policy-gradient loss, configured once for both objectives that train it: online GRPO
//! ([`super::grpo::GrpoConfig::loss`], sampling from an environment) and the recorded-rollout objective
//! ([`super::external::PolicyGradientConfig::loss`], reading records). What differs between the two, where the rows
//! come from, stays in each objective; the band, the KL term, the aggregation, the advantages and the step guards are
//! [`PgLoss`].

use model::Model;

use super::advantage::{AdvantageFloor, AdvantageNorm};
use super::grpo::LossAgg;
use super::grpo_term::ClipBand;
use super::pg_batch::PgSpec;

/// Consecutive optimiser steps the step guard may skip before the run is aborted.
pub const MAX_CONSECUTIVE_GUARD_SKIPS: u32 = 8;

/// The clipped policy-gradient loss of one step, and the bounds a step must keep to reach the weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PgLoss {
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
    /// Coefficient of the k3 KL to the reference policy; `0.0` puts no KL term in the loss.
    /// The term is k3 weighted by the importance ratio to the behaviour policy, so its value and its expected
    /// gradient are those of `KL(pi || ref)`, the divergence reported (`grpo_term::token_term_weighted`).
    pub kl_beta: f32,
    /// Measure and report the KL to the reference even when `kl_beta` is zero.
    pub log_kl: bool,
    /// How token terms are normalised across the rows of a step.
    pub loss_agg: LossAgg,
    /// How each group's rewards become the advantages of its rollouts.
    pub advantage_norm: AdvantageNorm,
    /// What is done with those advantages once computed against the whole group ([`AdvantageFloor::Zero`]: a rollout
    /// under the baseline sets it but carries no policy gradient; the KL term, when on, still reaches its tokens).
    pub advantage_floor: AdvantageFloor,
    /// Skip, and log, the optimiser step of a step whose mean KL to the reference (the step's `kl`) exceeds this: a
    /// rare step with a few extreme log-ratios (KL 0.50 among neighbours of 0.003) must not reach the weights. Needs
    /// a reference measurement ([`Self::log_kl`] or a KL weight). `None`: no bound.
    pub max_step_kl: Option<f32>,
    /// Skip, and log, the optimiser step of a step whose global gradient norm (after averaging over the step's
    /// micro-batches, before the optimiser's clip) exceeds this. `None`: no bound.
    pub max_step_grad_norm: Option<f32>,
}

/// A loss setting no run can honour: the field, its value and why.
#[derive(Clone, Debug, PartialEq)]
pub struct LossRefusal {
    pub field: &'static str,
    pub value: f32,
    pub why: String,
}

impl std::fmt::Display for LossRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.why)
    }
}

impl PgLoss {
    /// The recorded-rollout objective's recommended loss: GRPO's published aggregation (each sequence's mean), a
    /// symmetric band with the dual clip, a small KL to the reference.
    pub const RECORDED: PgLoss = PgLoss { clip_eps: 0.2, clip_eps_high: None, dual_clip: Some(DEFAULT_DUAL_CLIP), kl_beta: 0.02, log_kl: true, loss_agg: LossAgg::SeqMean, advantage_norm: AdvantageNorm::MeanCentered, advantage_floor: AdvantageFloor::None, max_step_kl: None, max_step_grad_norm: None };

    /// Online GRPO's recommended loss: clip-higher (`[0.8, 1.28]`), every token of a step weighed alike, no KL term,
    /// no dual clip and no step guard (each is a setting of its own).
    pub const ONLINE: PgLoss = PgLoss { clip_eps: 0.2, clip_eps_high: Some(0.28), dual_clip: None, kl_beta: 0.0, log_kl: true, loss_agg: LossAgg::TokenSum, advantage_norm: AdvantageNorm::MeanCentered, advantage_floor: AdvantageFloor::None, max_step_kl: None, max_step_grad_norm: None };

    /// The clip band, with the dual clip when one is set.
    pub fn band(&self) -> ClipBand {
        ClipBand { low: self.clip_eps, high: self.clip_eps_high.unwrap_or(self.clip_eps), dual: self.dual_clip }
    }

    /// The clipped surrogate over [`Self::band`] with the KL weight.
    pub fn spec(&self) -> PgSpec {
        PgSpec::clipped(self.band(), self.kl_beta)
    }

    /// Whether the run measures the KL to the reference: a KL term, or a KL logged without one.
    pub fn wants_reference(&self) -> bool {
        self.kl_beta > 0.0 || self.log_kl
    }

    /// Refuse a setting that has no meaning.
    ///
    /// # Errors
    /// The first setting out of its range, by field.
    pub fn validate(&self) -> Result<(), LossRefusal> {
        let refuse = |field: &'static str, value: f32, why: String| Err(LossRefusal { field, value, why });
        if let Some(c) = self.dual_clip.filter(|c| !(c.is_finite() && *c > 1.0)) {
            return refuse("dual_clip", c, format!("dual_clip is {c}; it must be a finite number above 1 (leave it unset for the plain clipped surrogate)"));
        }
        if !(self.clip_eps.is_finite() && self.clip_eps > 0.0 && self.clip_eps < 1.0) {
            return refuse("clip_eps", self.clip_eps, format!("clip_eps is {}; it must lie in (0, 1)", self.clip_eps));
        }
        if let Some(high) = self.clip_eps_high.filter(|h| !(h.is_finite() && *h >= self.clip_eps && *h < 10.0)) {
            return refuse("clip_eps_high", high, format!("clip_eps_high is {high}; it must be finite, at least clip_eps ({}) and below 10 (leave it unset for a symmetric band)", self.clip_eps));
        }
        if !(self.kl_beta.is_finite() && self.kl_beta >= 0.0) {
            return refuse("kl_beta", self.kl_beta, format!("kl_beta is {}; it must be a finite number, at least 0", self.kl_beta));
        }
        for (field, value) in [("max_step_kl", self.max_step_kl), ("max_step_grad_norm", self.max_step_grad_norm)] {
            if let Some(v) = value.filter(|v| !(v.is_finite() && *v > 0.0)) {
                return refuse(field, v, format!("{field} is {v}; it must be finite and positive (leave it unset for no guard)"));
            }
        }
        if let Some(bound) = self.max_step_kl.filter(|_| !self.wants_reference()) {
            return refuse("max_step_kl", bound, "max_step_kl bounds the KL to the reference, which is measured only with log_kl or a positive kl_beta".to_string());
        }
        if let LossAgg::ConstantLength(0) = self.loss_agg {
            return refuse("loss_agg", 0.0, LossAgg::ConstantLength(0).validate().expect_err("a zero constant length is refused"));
        }
        Ok(())
    }

    /// Whether any step guard is set.
    pub fn guards_steps(&self) -> bool {
        self.max_step_kl.is_some() || self.max_step_grad_norm.is_some()
    }
}

/// The dual clip the recorded-rollout objective defaults to: the `c = 3` of dual-clip PPO.
pub const DEFAULT_DUAL_CLIP: f32 = 3.0;

/// What the step guard says about a step that was finite.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum GuardVerdict {
    Pass,
    /// Over a bound: the step is skipped, and why.
    Skip(String),
    /// Over a bound for the [`MAX_CONSECUTIVE_GUARD_SKIPS`]th step in a row: the run stops. Why, and the run's error.
    Abort { why: String, error: String },
}

/// The step guard's count of the steps it skipped in a row.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct StepGuard {
    consecutive: u32,
}

impl StepGuard {
    /// The verdict on a finite step whose mean KL to the reference was `kl` (`None`: not measured, which no bound
    /// judges) and whose gradient norm was `grad_norm` (`None`: not read, there being no bound on it).
    pub(crate) fn judge(&mut self, loss: &PgLoss, kl: Option<f32>, grad_norm: Option<f32>) -> GuardVerdict {
        let over_kl = loss.max_step_kl.zip(kl).filter(|(bound, kl)| kl > bound);
        let over_grad = loss.max_step_grad_norm.zip(grad_norm).filter(|(bound, norm)| norm > bound);
        let norm = grad_norm.map_or_else(String::new, |n| format!(" (gradient norm {n:.3})"));
        let why = match (over_kl, over_grad) {
            (Some((bound, kl)), _) => format!("the step's KL {kl:.4} is over the bound {bound}{norm}"),
            (None, Some((bound, norm))) => format!("the step's gradient norm {norm:.3} is over the bound {bound}"),
            (None, None) => {
                self.consecutive = 0;
                return GuardVerdict::Pass;
            }
        };
        self.consecutive += 1;
        if self.consecutive >= MAX_CONSECUTIVE_GUARD_SKIPS {
            let error = format!("{MAX_CONSECUTIVE_GUARD_SKIPS} optimiser steps in a row were skipped by the step guard ({why}); the bound is too tight for this policy, or the policy has diverged");
            return GuardVerdict::Abort { why, error };
        }
        GuardVerdict::Skip(why)
    }

    /// A step that was not judged (a non-finite one) breaks the run of skips.
    pub(crate) fn reset(&mut self) {
        self.consecutive = 0;
    }
}

/// Whether every gradient of the model is finite, and the sum of their squares (in f64), read back once. A model that
/// does not expose its gradients counts as finite with a zero norm.
pub(crate) fn gradient_scan<M: Model>(model: &M) -> (bool, f64) {
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

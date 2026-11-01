// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for reward-driven training that reduces to one gradient-checked weighted
// cross-entropy kernel, for its clients. If your team needs expertise in policy-gradient fine-tuning of quantized
// language models then you can procure our services by sending an email to info@swedishembedded.com.

//! One batch of the clipped policy-gradient objective: the single implementation both [`super::grpo::Grpo`] (rows it
//! sampled itself) and [`super::external::ExternalPolicyGradient`] (rows collected elsewhere) train through.
//!
//! The caller has set a batch on the model and run [`model::Model::forward`]. [`backward_weighted`] then reads the
//! forward's per-token log-probabilities, turns them into per-token weights with [`super::grpo_term::token_term_banded`]
//! (the clipped surrogate and the k3 KL, one function), installs the weights and runs the backward. A token whose
//! advantage is NaN has none: it is an imitation token, trained with the plain cross-entropy weight `1.0` and no KL
//! (what a supervised record in a mixed dataset is).

use model::training;
use model::{Model, IGNORE};

use super::grpo_term::{k3_kl, token_term_banded, ClipBand};

/// The objective's two knobs besides the data: the clip band and the weight of the KL term.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PgSpec {
    pub band: ClipBand,
    /// k3 KL-to-reference coefficient; `0.0` puts no KL term in the loss.
    pub kl_beta: f32,
}

/// The per-position inputs of one batch, flat over the batch's `b * t` positions.
#[derive(Clone, Copy, Debug)]
pub struct PgBatch<'a> {
    /// The target at each position, [`IGNORE`] where nothing is supervised.
    pub targets: &'a [u32],
    /// The behaviour policy's log-probability of each target.
    pub old_lp: &'a [f32],
    /// The reference policy's log-probability of each target, when one was measured.
    pub ref_lp: Option<&'a [f32]>,
    /// The advantage of each position's sequence; NaN marks an imitation position.
    pub advantage: &'a [f32],
}

/// What one batch measured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BatchTerms {
    /// The sum of the token losses (the negative surrogate, plus the KL term, plus the cross-entropy of imitation
    /// tokens).
    pub loss: f32,
    pub supervised: usize,
    /// Tokens trained by the policy-gradient term (those with an advantage).
    pub policy_tokens: usize,
    /// Of those, the tokens with a non-zero advantage whose ratio sat outside the clip band on the side the advantage
    /// pushes toward, so they carry no gradient.
    pub clipped: usize,
    /// Tokens with a non-zero advantage (what [`Self::clipped`] is a fraction of).
    pub signed_tokens: usize,
    pub imitation_tokens: usize,
    /// The sum of the k3 KL to the reference over the policy tokens, and how many tokens it covers.
    pub kl_sum: f32,
    pub kl_tokens: usize,
    /// The importance ratio `exp(new - old)` over the policy tokens: its sum, smallest and largest value.
    pub ratio_sum: f64,
    pub ratio_min: f32,
    pub ratio_max: f32,
    /// The largest `|ratio - 1|` over the policy tokens.
    pub max_ratio_dev: f32,
    /// Every log-probability, weight and loss was finite.
    pub finite: bool,
}

impl Default for BatchTerms {
    fn default() -> BatchTerms {
        BatchTerms {
            loss: 0.0,
            supervised: 0,
            policy_tokens: 0,
            clipped: 0,
            signed_tokens: 0,
            imitation_tokens: 0,
            kl_sum: 0.0,
            kl_tokens: 0,
            ratio_sum: 0.0,
            ratio_min: f32::INFINITY,
            ratio_max: f32::NEG_INFINITY,
            max_ratio_dev: 0.0,
            finite: true,
        }
    }
}

/// The per-position CE-gradient weights for `new_lp` (the forward's log-probabilities) and what they measured.
pub fn weigh(new_lp: &[f32], batch: &PgBatch<'_>, spec: &PgSpec) -> (Vec<f32>, BatchTerms) {
    let mut weights = vec![0f32; new_lp.len()];
    let mut terms = BatchTerms::default();
    for i in (0..new_lp.len()).filter(|&i| batch.targets[i] != IGNORE) {
        terms.supervised += 1;
        terms.finite &= new_lp[i].is_finite();
        let advantage = batch.advantage[i];
        if advantage.is_nan() {
            weights[i] = 1.0;
            terms.loss -= new_lp[i];
            terms.imitation_tokens += 1;
            continue;
        }
        let ref_lp = batch.ref_lp.map(|r| r[i]);
        let term = token_term_banded(batch.old_lp[i], new_lp[i], advantage, spec.band, ref_lp.filter(|_| spec.kl_beta > 0.0), spec.kl_beta);
        weights[i] = term.weight;
        terms.loss += term.loss;
        terms.policy_tokens += 1;
        terms.signed_tokens += usize::from(advantage != 0.0);
        terms.clipped += usize::from(term.clipped);
        if let Some(r) = ref_lp {
            terms.kl_sum += k3_kl(r, new_lp[i]);
            terms.kl_tokens += 1;
        }
        let ratio = (new_lp[i] - batch.old_lp[i]).exp();
        terms.ratio_sum += f64::from(ratio);
        terms.ratio_min = terms.ratio_min.min(ratio);
        terms.ratio_max = terms.ratio_max.max(ratio);
        terms.max_ratio_dev = terms.max_ratio_dev.max((ratio - 1.0).abs());
    }
    terms.finite &= terms.loss.is_finite() && weights.iter().all(|w| w.is_finite());
    (weights, terms)
}

/// Backward the objective of the batch just forwarded. `denominator` replaces the batch's own supervised-token count
/// as the divisor of loss and gradient ([`training::TokenObjectives::set_loss_denominator`]); `None` keeps the
/// per-batch mean. A batch with a non-finite log-probability, weight or loss is not differentiated (`finite` is
/// `false`) and leaves the gradients as they were.
pub fn backward_weighted<M: Model>(model: &M, batch: &PgBatch<'_>, spec: &PgSpec, denominator: Option<f32>) -> BatchTerms {
    let objectives = training::objectives(model);
    if let Some(d) = denominator {
        objectives.set_loss_denominator(d);
    }
    let (weights, terms) = weigh(&objectives.batch_token_logprobs(), batch, spec);
    if terms.finite {
        objectives.set_loss_weights(&weights);
        model.backward();
    }
    terms
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPEC: PgSpec = PgSpec { band: ClipBand { low: 0.2, high: 0.2 }, kl_beta: 0.0 };

    fn batch<'a>(targets: &'a [u32], old: &'a [f32], advantage: &'a [f32]) -> PgBatch<'a> {
        PgBatch { targets, old_lp: old, ref_lp: None, advantage }
    }

    /// An imitation token weighs 1 and costs its cross-entropy, whatever its old log-probability.
    #[test]
    fn a_token_without_an_advantage_is_plain_cross_entropy() {
        let (w, t) = weigh(&[-0.7, -1.2], &batch(&[3, IGNORE], &[0.0, 0.0], &[f32::NAN, f32::NAN]), &SPEC);
        assert_eq!(w, [1.0, 0.0]);
        assert!((t.loss - 0.7).abs() < 1e-6 && t.imitation_tokens == 1 && t.policy_tokens == 0 && t.supervised == 1, "{t:?}");
    }

    /// The statistics count clipping only where an advantage was clipped.
    #[test]
    fn clip_fraction_counts_signed_tokens_outside_the_band_on_their_side() {
        let (new, old) = ([1.5f32.ln(), 1.5f32.ln(), 0.0, 0.0], [0.0; 4]);
        // Positive advantage past the upper edge: clipped. Negative advantage there: not. Zero advantage: unsigned.
        let (w, t) = weigh(&new, &batch(&[1, 1, 1, 1], &old, &[1.0, -1.0, 0.0, 1.0]), &SPEC);
        assert_eq!((t.signed_tokens, t.clipped), (3, 1), "{t:?}");
        assert_eq!(w[0], 0.0);
        assert!((w[1] + 1.5).abs() < 1e-5);
        assert!((t.ratio_max - 1.5).abs() < 1e-5 && (t.ratio_min - 1.0).abs() < 1e-6, "{t:?}");
    }

    #[test]
    fn a_non_finite_loss_is_reported_not_hidden() {
        let (_, t) = weigh(&[f32::NAN], &batch(&[1], &[0.0], &[1.0]), &SPEC);
        assert!(!t.finite);
    }
}

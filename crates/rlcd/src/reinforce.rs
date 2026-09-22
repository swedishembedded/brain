// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! RLCD's actual update rule: REINFORCE over [`crate::proper`]'s reward, with
//! a group-mean baseline.
//!
//! This is the objective the `convaiinnovations/laya` release is trained
//! with, transcribed from the only PUBLIC Laya training loop - the DDP
//! fine-tuning script embedded in the project's own
//! `laya_finetune_typed_decisions_2xT4_kaggle.ipynb`
//! (github.com/NandhaKishorM/laya) - cross-read against the model card's
//! prose ("the reward is a strictly proper scoring rule ... updates are
//! REINFORCE with a group-mean baseline (GRPO-style)").
//!
//! ## The shape of one update
//!
//! For one question's raw option scores `z` (`k` options) and target `t`:
//!
//! 1. Draw `G` Gaussian exploration vectors `eps_g ~ N(0, sigma^2 I)` and
//!    project each to zero mean across the options. The projection is not
//!    cosmetic: softmax is shift-invariant, so the component of the noise
//!    along `(1, 1, ..., 1)` changes no probability and would be pure
//!    variance in the gradient estimate.
//! 2. Score each sampled report `q_g = softmax(z + eps_g)` with
//!    [`crate::proper::proper_reward`], **without a gradient** - the reward
//!    is a black box, exactly as the reference computes it under
//!    `torch.no_grad()`.
//! 3. Standardize into an advantage, `adv = (r - mean(r)) / (std(r) + 1e-6)`
//!    (the group-mean baseline, GRPO-style).
//! 4. The policy term is `-mean_g(adv_g * log p(z + eps_g | z))` under the
//!    Gaussian exploration density, whose only `z`-dependence is
//!    `-||(z + eps_g) - z||^2 / (2 sigma^2)`. The sampled point is a
//!    constant; `z` is not.
//! 5. Add a soft cross-entropy term at weight [`RlcdObjective::ce_weight`].
//!
//! ## What is pinned down and what is a choice
//!
//! Pinned down by the public fine-tuning loop: this five-step shape, the
//! `1e-6` std guard, the zero-mean projection, `w_rps = 1.0`, and its own
//! settings `G = 4`, `sigma: 0.4 -> 0.1`, `w_sph = 0.75`, `ce_weight = 1.0`.
//! The model card's prose describes the ORIGINAL pretraining run differently
//! (`G = 8`, `sigma: 1.0 -> 0.3`, `w_sph = 0.5`, and "pure policy gradient
//! (zero supervised cross-entropy loss)", i.e. `ce_weight = 0.0`); that run's
//! script was never published. [`RlcdObjective::default`] therefore takes the
//! settings of the loop that EXISTS in source, and every one of them is a
//! public dial rather than a constant, so the pretraining settings are
//! reachable too ([`RlcdObjective::pretrain`]).
//!
//! **Not implemented here, because no public source pins it down**: the
//! act/escalate head's own cost-sensitive objective. The published loop
//! deliberately does not train it (its only act-head term is a `0.0 * act`
//! no-op that exists to give DDP a gradient path), and the cost matrix behind
//! `rl_agent_config.json`'s `act_costs`/`cost_wrong_act` appears in prose
//! only. A caller therefore gets a zero act-logit gradient, which leaves that
//! head exactly as the checkpoint shipped it.
//!
//! ## One question per call
//!
//! The reference standardizes the advantage over its whole micro-batch
//! (`adv.std()` with no `dim`) and averages the policy term over it. This
//! module scores ONE question per call, which is the granularity this
//! workspace's decision training contract works at; at a micro-batch of one
//! question the two are the same arithmetic. A caller batching several
//! questions accumulates several of these and scales the optimizer step,
//! which is a different (and lower-variance) normalization than the
//! reference's - stated here rather than hidden.
//!
//! Swedish Embedded AB trains decision models against the reward that
//! actually matters to the operator rather than against a convenient
//! surrogate. If your team needs expertise in policy-gradient training of
//! calibrated decision models, you can procure our services by sending an
//! email to info@swedishembedded.com.

use crate::proper::{proper_reward, ProperScore};

/// The dials of one RLCD update. See the module doc for which values come
/// from which published run.
#[derive(Clone, Copy, Debug)]
pub struct RlcdObjective {
    /// `G`, the number of exploration samples whose mean reward is the
    /// baseline. Must be at least 2 - a group of one has no baseline and a
    /// zero advantage.
    pub group: usize,
    /// Exploration standard deviation, in logit units. Annealed over a run
    /// by [`anneal`].
    pub sigma: f32,
    /// Weight on the auxiliary soft cross-entropy term. `0.0` is the model
    /// card's "pure policy gradient".
    pub ce_weight: f32,
    /// The reward the samples are scored with.
    pub score: ProperScore,
}

impl Default for RlcdObjective {
    /// The published fine-tuning loop's own settings - see the module doc.
    fn default() -> RlcdObjective {
        RlcdObjective {
            group: 4,
            sigma: 0.4,
            ce_weight: 1.0,
            score: ProperScore {
                w_sph: 0.75,
                ..ProperScore::default()
            },
        }
    }
}

impl RlcdObjective {
    /// The settings the model card attributes to the ORIGINAL pretraining
    /// run, whose script was never published: `G = 8`, `sigma` from `1.0`,
    /// `w_sph = 0.5`, and no cross-entropy term at all.
    pub fn pretrain() -> RlcdObjective {
        RlcdObjective {
            group: 8,
            sigma: 1.0,
            ce_weight: 0.0,
            score: ProperScore::default(),
        }
    }

    /// The exploration scale at `progress` through a run, on `[0, 1]`.
    pub fn at(mut self, sigma: f32) -> RlcdObjective {
        self.sigma = sigma;
        self
    }
}

/// Linear anneal from `start` to `end` at `progress` in `[0, 1]` - the shape
/// both published runs use for `sigma` (`0.4 -> 0.1` fine-tuning,
/// `1.0 -> 0.3` pretraining). Clamped, so a caller that overruns its own step
/// budget gets `end` rather than an extrapolation.
pub fn anneal(start: f32, end: f32, progress: f32) -> f32 {
    let p = progress.clamp(0.0, 1.0);
    start + (end - start) * p
}

/// The published loop's learning-rate schedule: cosine annealing from `base`
/// to `min` with NO warmup, at `progress` in `[0, 1]`.
pub fn cosine_lr(base: f32, min: f32, progress: f32) -> f32 {
    let p = progress.clamp(0.0, 1.0);
    min + 0.5 * (base - min) * (1.0 + (std::f32::consts::PI * p).cos())
}

/// One group of exploration samples and the advantage each earned - step 1
/// through 3 of the module doc, i.e. everything the reference computes under
/// `torch.no_grad()`.
///
/// `samples[g]` is the sampled score vector `z + eps_g` (not the noise), so
/// [`policy_loss`] needs nothing else to reconstruct the exploration density.
#[derive(Clone, Debug)]
pub struct Group {
    pub samples: Vec<Vec<f32>>,
    pub advantage: Vec<f32>,
    /// The `sigma` the samples were actually drawn at, carried alongside them
    /// so a caller cannot pair a group with the wrong exploration density.
    pub sigma: f32,
}

/// Draw one [`Group`]: `cfg.group` zero-mean-projected Gaussian perturbations
/// of `scores`, each scored by [`proper_reward`], standardized into an
/// advantage.
pub fn explore(
    scores: &[f32],
    target: &[f32],
    ordinal: bool,
    cfg: &RlcdObjective,
    rng: &mut data::rng::Rng,
) -> Group {
    assert!(cfg.group >= 2, "a group baseline needs at least 2 samples, got {}", cfg.group);
    assert_eq!(scores.len(), target.len(), "scores and target must have the same arity");
    let k = scores.len();
    let mut samples = Vec::with_capacity(cfg.group);
    let mut rewards = Vec::with_capacity(cfg.group);
    for _ in 0..cfg.group {
        let mut eps: Vec<f32> = (0..k).map(|_| rng.next_gaussian() as f32 * cfg.sigma).collect();
        // Zero-mean projection: softmax is shift-invariant, so the component
        // along (1, ..., 1) is noise that cannot change any reported
        // probability. Applied AFTER the sigma scaling, matching the
        // reference's own order (which very slightly shrinks the realized
        // variance, and reproducing it is the point).
        let mean = eps.iter().map(|&v| v as f64).sum::<f64>() / k as f64;
        for e in &mut eps {
            *e -= mean as f32;
        }
        let z: Vec<f32> = scores.iter().zip(&eps).map(|(&s, &e)| s + e).collect();
        rewards.push(proper_reward(&crate::scoring::softmax(&z), target, ordinal, &cfg.score));
        samples.push(z);
    }
    Group {
        advantage: standardize(&rewards),
        samples,
        sigma: cfg.sigma,
    }
}

/// `(r - mean(r)) / (std(r) + 1e-6)`, with torch's own UNBIASED (`ddof = 1`)
/// standard deviation - `Tensor::std`'s default, and the one the reference
/// therefore uses.
fn standardize(r: &[f32]) -> Vec<f32> {
    let n = r.len() as f64;
    let mean = r.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = r.iter().map(|&v| (v as f64 - mean) * (v as f64 - mean)).sum::<f64>() / (n - 1.0).max(1.0);
    let denom = var.sqrt() + 1e-6;
    r.iter().map(|&v| ((v as f64 - mean) / denom) as f32).collect()
}

/// The DIFFERENTIABLE half - steps 4 and 5 - given a group that is already
/// fixed. Returns the loss and `dL/d(score)`.
///
/// Split out from [`explore`] on exactly the boundary the reference's
/// `torch.no_grad()` draws, which is also what makes this half
/// finite-difference checkable: its samples and advantages are constants, so
/// perturbing `scores` here reproduces the same function autograd
/// differentiates.
pub fn policy_loss(
    scores: &[f32],
    target: &[f32],
    group: &Group,
    ce_weight: f32,
) -> (f32, Vec<f32>) {
    let k = scores.len();
    assert_eq!(target.len(), k, "scores and target must have the same arity");
    assert!(!group.samples.is_empty(), "an empty group has no policy gradient");
    assert_eq!(group.samples.len(), group.advantage.len(), "one advantage per sample");
    let g = group.samples.len() as f64;
    let two_sigma_sq = 2.0 * group.sigma as f64 * group.sigma as f64;
    assert!(two_sigma_sq > 0.0, "sigma must be positive, got {}", group.sigma);

    // --- policy term: -mean_g(adv_g * logp_g), logp_g = -||z_g - z||^2 / 2s^2
    let mut loss_rl = 0.0f64;
    let mut d = vec![0.0f64; k];
    for (z, &adv) in group.samples.iter().zip(&group.advantage) {
        assert_eq!(z.len(), k, "a sample has a different arity than the scores");
        let sq: f64 = z
            .iter()
            .zip(scores)
            .map(|(&zi, &si)| (zi as f64 - si as f64) * (zi as f64 - si as f64))
            .sum();
        loss_rl += adv as f64 * sq / two_sigma_sq / g;
        // d/ds_i of [-adv * -(z_i - s_i)^2 / 2s^2] / G = -adv (z_i - s_i)/(s^2 G)
        for i in 0..k {
            d[i] -= adv as f64 * (z[i] as f64 - scores[i] as f64) * 2.0 / two_sigma_sq / g;
        }
    }

    // --- soft cross-entropy: -sum_i t_i * log_softmax(z)_i ---
    let p = crate::scoring::softmax(scores);
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = max + scores.iter().map(|&s| (s as f64 - max).exp()).sum::<f64>().ln();
    let t_sum: f64 = target.iter().map(|&t| t as f64).sum();
    let mut loss_ce = 0.0f64;
    for i in 0..k {
        loss_ce -= target[i] as f64 * (scores[i] as f64 - lse);
    }
    let w = ce_weight as f64;
    for i in 0..k {
        d[i] += w * (t_sum * p[i] as f64 - target[i] as f64);
    }

    ((loss_rl + w * loss_ce) as f32, d.into_iter().map(|v| v as f32).collect())
}

/// One complete RLCD objective evaluation: [`explore`] then [`policy_loss`].
///
/// The `(loss, dL/d(score))` pair is the same shape
/// [`crate::scoring::decision_loss_soft`] returns, so a training loop can
/// swap one objective for the other without changing anything downstream.
pub fn rlcd_loss(
    scores: &[f32],
    target: &[f32],
    ordinal: bool,
    cfg: &RlcdObjective,
    rng: &mut data::rng::Rng,
) -> (f32, Vec<f32>) {
    let group = explore(scores, target, ordinal, cfg, rng);
    policy_loss(scores, target, &group, cfg.ce_weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proper::hard_target;
    use crate::scoring::softmax;

    fn fixed_group(sigma: f32) -> Group {
        Group {
            samples: vec![
                vec![0.61, -1.05, 2.22, 0.40],
                vec![0.19, -1.44, 1.81, -0.11],
                vec![0.35, -0.98, 2.05, 0.21],
            ],
            advantage: vec![1.21, -0.77, -0.44],
            sigma,
        }
    }

    /// The gradient is hand-derived through the exploration density and a
    /// softmax, which is exactly the kind of derivation that is plausibly
    /// wrong. Checked on the DIFFERENTIABLE half with the group held fixed -
    /// the same function autograd sees on the other side of the reference's
    /// own `torch.no_grad()`.
    #[test]
    fn the_policy_gradient_matches_finite_differences() {
        let scores = [0.4f32, -1.2, 2.0, 0.1];
        let targets: [[f32; 4]; 2] = [[0.0, 0.0, 1.0, 0.0], [0.4, 0.1, 0.3, 0.2]];
        for ce_weight in [0.0f32, 1.0] {
            for target in &targets {
                let group = fixed_group(0.4);
                let (_, d) = policy_loss(&scores, target, &group, ce_weight);
                for i in 0..scores.len() {
                    let eps = 1e-3f32;
                    let (mut up, mut dn) = (scores, scores);
                    up[i] += eps;
                    dn[i] -= eps;
                    let num = (policy_loss(&up, target, &group, ce_weight).0
                        - policy_loss(&dn, target, &group, ce_weight).0)
                        / (2.0 * eps);
                    let tol = 2e-3 + 2e-2 * d[i].abs().max(num.abs());
                    assert!(
                        (d[i] - num).abs() <= tol,
                        "ce_weight {ce_weight} target {target:?} i {i}: analytic {} vs numeric {num}",
                        d[i]
                    );
                }
            }
        }
    }

    /// At `ce_weight = 0` the whole objective is the policy term, and its
    /// gradient must NOT sum to zero the way a softmax-normalized loss's
    /// does - the exploration density is over raw logits, not probabilities.
    /// At `ce_weight > 0` the cross-entropy half alone does sum to zero.
    /// Checking both is what stops the two halves from being silently
    /// conflated.
    #[test]
    fn the_cross_entropy_half_is_shift_invariant_and_the_policy_half_is_not() {
        let scores = [0.4f32, -1.2, 2.0, 0.1];
        let t = hard_target(4, 2);
        let group = fixed_group(0.4);
        let (_, d_ce_only) = {
            let empty = Group { samples: vec![vec![0.0; 4]], advantage: vec![0.0], sigma: 0.4 };
            policy_loss(&scores, &t, &empty, 1.0)
        };
        let sum_ce: f32 = d_ce_only.iter().sum();
        assert!(sum_ce.abs() <= 1e-5, "the cross-entropy gradient sums to {sum_ce}, not 0");

        let (_, d_rl) = policy_loss(&scores, &t, &group, 0.0);
        let sum_rl: f32 = d_rl.iter().sum();
        assert!(
            sum_rl.abs() > 1e-3,
            "the policy gradient sums to {sum_rl}: it is behaving like a normalized loss"
        );
    }

    /// The exploration noise must be zero-mean ACROSS OPTIONS on every single
    /// sample, not merely in expectation - that projection is what keeps the
    /// estimator from spending its variance on a direction softmax ignores.
    #[test]
    fn every_exploration_sample_is_zero_mean_across_the_options() {
        let scores = [0.4f32, -1.2, 2.0, 0.1, 0.9];
        let t = hard_target(5, 1);
        let cfg = RlcdObjective::default();
        let mut rng = data::rng::Rng::new(0xD1CE_2026);
        let group = explore(&scores, &t, false, &cfg, &mut rng);
        assert_eq!(group.samples.len(), cfg.group);
        for z in &group.samples {
            let shift: f32 = z.iter().zip(&scores).map(|(&zi, &si)| zi - si).sum();
            assert!(shift.abs() <= 1e-4, "sample noise sums to {shift}, not 0");
        }
    }

    /// The baseline is the whole reason this is REINFORCE-with-a-baseline and
    /// not plain REINFORCE: advantages within a group must sum to zero, so a
    /// question whose samples are all equally good produces no update at all.
    #[test]
    fn the_group_advantage_is_centred() {
        let cfg = RlcdObjective::default();
        let mut rng = data::rng::Rng::new(7);
        let t = hard_target(3, 0);
        let group = explore(&[0.2f32, 1.0, -0.4], &t, false, &cfg, &mut rng);
        let sum: f32 = group.advantage.iter().sum();
        assert!(sum.abs() <= 1e-4, "advantages sum to {sum}, not 0");
    }

    /// The point of the whole construction: following the negative gradient
    /// must raise the reward that was actually sampled. Verified end to end on
    /// a bare score vector (no model), by taking real gradient steps and
    /// watching [`proper_reward`] on the reported distribution improve toward
    /// its optimum - which for a strictly proper rule is the target itself.
    #[test]
    fn descending_the_objective_raises_the_proper_reward() {
        let cfg = RlcdObjective {
            ce_weight: 0.0, // the pure policy gradient - no supervised help
            ..RlcdObjective::default()
        };
        let target = [0.55f32, 0.25, 0.15, 0.05];
        let mut rng = data::rng::Rng::new(0x5EED_2026);
        let mut z = vec![0.0f32; 4];
        let before = proper_reward(&softmax(&z), &target, false, &cfg.score);
        for _ in 0..4000 {
            let (_, d) = rlcd_loss(&z, &target, false, &cfg, &mut rng);
            for (zi, &di) in z.iter_mut().zip(&d) {
                *zi -= 0.02 * di;
            }
        }
        let after = proper_reward(&softmax(&z), &target, false, &cfg.score);
        let best = proper_reward(&target, &target, false, &cfg.score);
        assert!(
            after > before,
            "the policy gradient did not improve the reward: {before} -> {after}"
        );
        // Strictly proper: the optimum IS the target, so a working policy
        // gradient must get close to it, not merely somewhere better.
        let p = softmax(&z);
        let l1: f32 = p.iter().zip(&target).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            l1 < 0.10,
            "converged to {p:?}, L1 {l1} away from the target {target:?} (reward {after} vs optimum {best})"
        );
    }
}

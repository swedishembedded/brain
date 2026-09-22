// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Searching a [`crate::atlas::World`] for cases where a learner's induced
//! action disagrees with the oracle's, and expanding each one into a
//! corrective family.
//!
//! A bare failing example is weak supervision: it tells a learner an answer
//! was wrong, not why, and not what would have made it right. This module
//! searches a world's observations (plus its own prior, the "no evidence yet"
//! point) for exactly the cases where a learner's belief leads it to the
//! WRONG action under an explicit cost matrix, scored by true regret against
//! the oracle - never by probability distance alone, since two beliefs that
//! differ numerically but agree on every action of interest are not a
//! decision failure. Each hit is then expanded into five related points: the
//! failing observation itself, a nearby one the learner gets right, the
//! evidence reveal that resolves the ambiguity (when one exists), a cost
//! regime from the caller's own bounded sweep that would have changed the
//! oracle's OWN answer, and a relabeled copy proving the failure is anchored
//! to evidence rather than wording.
//!
//! The world here is small and enumerable, so this is exhaustive search over
//! its observations and the caller's supplied cost sweep - not approximate
//! optimization over a continuous space. That is what lets a found witness be
//! a certificate (every candidate was checked) rather than a sample (some
//! candidates might have been missed).

use crate::atlas::{Distribution, Observation, World};
use crate::cost::{bayes_action, regret, CostMatrix};

/// A predictor whose calibration is being AUDITED - not itself part of the
/// oracle. [`search`] feeds its output through the exact same
/// [`crate::cost::bayes_action`] the oracle's own posterior goes through;
/// the whole point of this module is to find where that substitution
/// produces a different action.
pub trait Learner {
    /// `&mut self`, not `&self`: a real trained model (device dispatches, KV
    /// or scratch buffers) generally needs mutable access to score anything.
    /// A pure-math learner (this module's own test fixtures included) can
    /// still implement this trivially without mutating.
    fn predict(&mut self, observation: &Observation) -> Distribution;
}

/// One learner failure, expanded into a corrective family. See this module's
/// doc for why a bare failing case is not enough supervision on its own.
#[derive(Clone, Debug)]
pub struct WitnessFamily {
    /// The observation where the learner's induced action differs from the
    /// oracle's - either one of the world's own observations, or its prior
    /// (labeled `"no evidence yet"`, representing acting before any query).
    pub failing: Observation,
    pub oracle_action: usize,
    pub learner_action: usize,
    /// True regret of the learner's action under the oracle's OWN belief at
    /// `failing` - always `> 0` for a member of this family.
    pub regret: f32,
    /// Another candidate point where the learner agrees with the oracle,
    /// chosen as the one closest to `failing` by posterior L2 distance.
    /// `None` only if every other candidate point is ALSO a failure.
    pub nearby_correct: Option<Observation>,
    /// If `failing` is the world's prior, the world's own observation whose
    /// posterior is FARTHEST from it - the single most decision-informative
    /// evidence reveal. `None` when `failing` is already a refined
    /// observation: this `World` exposes no finer partition to reveal.
    pub resolving_reveal: Option<Observation>,
    /// The first cost matrix in the caller's supplied sweep under which the
    /// ORACLE's own optimal action at `failing` differs from `oracle_action`,
    /// proving the failure is cost-sensitive and not merely probability-
    /// sensitive. `None` if nothing in the sweep flips it.
    pub cost_flip: Option<CostMatrix>,
    /// `failing`, relabeled with the identical posterior - proves the
    /// witness is anchored to evidence, not to wording.
    pub irrelevant_variation: Observation,
}

fn posterior_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| (x - y) * (x - y)).sum::<f32>().sqrt()
}

/// The world's prior, presented as a candidate decision point in its own
/// right - "act now, before running any query."
fn prior_as_observation(world: &impl World) -> Observation {
    Observation { name: "no evidence yet".into(), probability: 1.0, posterior: world.prior() }
}

fn candidate_points(world: &impl World) -> Vec<Observation> {
    let mut points = vec![prior_as_observation(world)];
    points.extend(world.observations());
    points
}

/// Exhaustively search `world`'s candidate points (its prior plus every
/// observation) for cases where `learner`'s induced action under `costs`
/// differs from the oracle's, and expand each into a [`WitnessFamily`].
/// `cost_sweep` bounds the search for [`WitnessFamily::cost_flip`] - it is
/// not itself searched for failures, only consulted per hit.
pub fn search(world: &impl World, learner: &mut impl Learner, costs: &CostMatrix, cost_sweep: &[CostMatrix]) -> Vec<WitnessFamily> {
    let points = candidate_points(world);
    let oracle_actions: Vec<usize> = points.iter().map(|p| bayes_action(&p.posterior, costs).action).collect();
    let learner_actions: Vec<usize> = points.iter().map(|p| bayes_action(&learner.predict(p), costs).action).collect();

    let mut families = Vec::new();
    for i in 0..points.len() {
        if oracle_actions[i] == learner_actions[i] {
            continue;
        }
        let failing = points[i].clone();
        let r = regret(&failing.posterior, costs, learner_actions[i]);

        let nearby_correct = (0..points.len())
            .filter(|&j| j != i && oracle_actions[j] == learner_actions[j])
            .min_by(|&a, &b| {
                posterior_distance(&failing.posterior, &points[a].posterior)
                    .partial_cmp(&posterior_distance(&failing.posterior, &points[b].posterior))
                    .expect("posterior distance is never NaN")
            })
            .map(|j| points[j].clone());

        let resolving_reveal = if failing.name == "no evidence yet" {
            world
                .observations()
                .into_iter()
                .max_by(|a, b| {
                    posterior_distance(&failing.posterior, &a.posterior)
                        .partial_cmp(&posterior_distance(&failing.posterior, &b.posterior))
                        .expect("posterior distance is never NaN")
                })
        } else {
            None
        };

        let cost_flip = cost_sweep.iter().find(|c| bayes_action(&failing.posterior, c).action != oracle_actions[i]).cloned();

        let irrelevant_variation =
            Observation { name: format!("{} (restated)", failing.name), probability: failing.probability, posterior: failing.posterior.clone() };

        families.push(WitnessFamily {
            failing,
            oracle_action: oracle_actions[i],
            learner_action: learner_actions[i],
            regret: r,
            nearby_correct,
            resolving_reveal,
            cost_flip,
            irrelevant_variation,
        });
    }
    families
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atlas::Distribution;

    /// The device-diagnosis world, restated as a private test fixture (see
    /// `crate::atlas`'s own tests for why concrete worlds are not published
    /// from this crate).
    struct DiagnosisWorld;

    impl World for DiagnosisWorld {
        fn n_outcomes(&self) -> usize {
            2
        }
        fn prior(&self) -> Distribution {
            vec![0.8, 0.2]
        }
        fn observations(&self) -> Vec<Observation> {
            let p_positive = 0.2 * 0.8 + 0.8 * 0.1;
            vec![
                Observation { name: "diagnostic: positive".into(), probability: p_positive, posterior: vec![1.0 / 3.0, 2.0 / 3.0] },
                Observation { name: "diagnostic: negative".into(), probability: 1.0 - p_positive, posterior: vec![18.0 / 19.0, 1.0 / 19.0] },
            ]
        }
    }

    /// Mirrors the oracle exactly - the null hypothesis the search must clear.
    struct PerfectLearner;
    impl Learner for PerfectLearner {
        fn predict(&mut self, observation: &Observation) -> Distribution {
            observation.posterior.clone()
        }
    }

    /// A deliberately planted miscalibration: reports 50/50 regardless of
    /// evidence. Its known failure point (worked out by hand below) is what
    /// the search's own result is checked against.
    struct ConstantHalfLearner;
    impl Learner for ConstantHalfLearner {
        fn predict(&mut self, _observation: &Observation) -> Distribution {
            vec![0.5, 0.5]
        }
    }

    const COSTS: fn() -> CostMatrix = || CostMatrix::binary(1.0, 10.0);

    #[test]
    fn the_oracle_itself_produces_zero_witnesses() {
        let families = search(&DiagnosisWorld, &mut PerfectLearner, &COSTS(), &[]);
        assert!(families.is_empty(), "a learner that mirrors the oracle exactly must never be flagged");
    }

    /// Hand-verified: under a constant 50/50 belief, the induced action is
    /// always "block" (expected cost 0.5 vs 5.0). The oracle's own action is
    /// also "block" at the prior and at a positive result, so those two
    /// points agree. Only at "diagnostic: negative" (`P(fault) = 1/19`,
    /// oracle action "release") does the constant learner disagree - exactly
    /// one witness, at a known location, with a computable regret.
    #[test]
    fn a_planted_miscalibration_is_found_exactly_where_expected() {
        let families = search(&DiagnosisWorld, &mut ConstantHalfLearner, &COSTS(), &[]);
        assert_eq!(families.len(), 1, "exactly one candidate point disagrees with a constant 50/50 belief");
        let w = &families[0];
        assert_eq!(w.failing.name, "diagnostic: negative");
        assert_eq!(w.oracle_action, 1, "release is optimal at P(fault) = 1/19");
        assert_eq!(w.learner_action, 0, "the constant learner always blocks");
        let expected_regret = 18.0 / 19.0 - 10.0 / 19.0; // cost(block) - cost(release) at this posterior
        assert!((w.regret - expected_regret).abs() <= 1e-5, "regret {} vs {expected_regret}", w.regret);
    }

    #[test]
    fn the_nearby_correct_point_is_the_closest_one_the_learner_still_gets_right() {
        let families = search(&DiagnosisWorld, &mut ConstantHalfLearner, &COSTS(), &[]);
        let nearby = families[0].nearby_correct.as_ref().expect("the prior and the positive result are both correct here");
        assert_eq!(nearby.name, "no evidence yet", "the prior (0.8, 0.2) is closer to (18/19, 1/19) than the positive result (1/3, 2/3) is");
    }

    #[test]
    fn irrelevant_variation_carries_the_identical_posterior() {
        let families = search(&DiagnosisWorld, &mut ConstantHalfLearner, &COSTS(), &[]);
        let w = &families[0];
        assert_eq!(w.irrelevant_variation.posterior, w.failing.posterior);
        assert_ne!(w.irrelevant_variation.name, w.failing.name, "the label must actually differ, or this proves nothing");
    }

    #[test]
    fn resolving_reveal_only_applies_when_the_failure_is_at_the_prior() {
        // Force the failure to be AT the prior: a learner that gets both
        // refined observations right but is wrong before any evidence arrives.
        struct WrongOnlyAtPrior;
        impl Learner for WrongOnlyAtPrior {
            fn predict(&mut self, observation: &Observation) -> Distribution {
                if observation.name == "no evidence yet" {
                    vec![0.5, 0.5] // induces "block", oracle says "block" too here - use a real flip below
                } else {
                    observation.posterior.clone()
                }
            }
        }
        // With C_FP raised so the PRIOR's oracle action is "release" (see
        // crate::cost's cost-sweep gate), a learner that still blocks at the
        // prior but is otherwise perfect fails ONLY at the prior.
        let costs = CostMatrix::binary(4.0, 10.0);
        let families = search(&DiagnosisWorld, &mut WrongOnlyAtPrior, &costs, &[]);
        assert_eq!(families.len(), 1);
        let w = &families[0];
        assert_eq!(w.failing.name, "no evidence yet");
        let reveal = w.resolving_reveal.as_ref().expect("a failure at the prior must offer a resolving reveal");
        assert_eq!(reveal.name, "diagnostic: positive", "the positive result (1/3, 2/3) is farther from the prior (0.8, 0.2) than the negative result (18/19, 1/19) is");

        // And the converse: a failure that is ALREADY a refined observation
        // has no finer partition in this World to reveal.
        let families2 = search(&DiagnosisWorld, &mut ConstantHalfLearner, &COSTS(), &[]);
        assert!(families2[0].resolving_reveal.is_none());
    }

    #[test]
    fn cost_flip_finds_a_swept_matrix_that_changes_the_oracles_own_action() {
        let sweep = vec![CostMatrix::binary(1.0, 100.0)]; // boundary 1/101, below 1/19: flips "negative" to "block"
        let families = search(&DiagnosisWorld, &mut ConstantHalfLearner, &COSTS(), &sweep);
        let flip = families[0].cost_flip.as_ref().expect("the swept matrix must flip the oracle's action at the failing point");
        assert_eq!(bayes_action(&families[0].failing.posterior, flip).action, 0, "block becomes optimal under the swept cost regime");
        assert_ne!(bayes_action(&families[0].failing.posterior, flip).action, families[0].oracle_action);
    }

    #[test]
    fn cost_flip_is_none_when_nothing_in_the_sweep_changes_the_action() {
        let sweep = vec![CostMatrix::binary(1.0, 10.0)]; // identical to COSTS() itself - cannot flip anything
        let families = search(&DiagnosisWorld, &mut ConstantHalfLearner, &COSTS(), &sweep);
        assert!(families[0].cost_flip.is_none());
    }
}

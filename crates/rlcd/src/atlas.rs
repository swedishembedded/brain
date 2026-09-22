// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Executable probabilistic worlds, and the contract a decision task is
//! defined under.
//!
//! A synthetic training example is only as trustworthy as the oracle that
//! labeled it. Three failure modes recur when a large model is simply asked
//! to invent a scenario, a label, and a confidence: the scenario, the label,
//! and the claimed certainty share one, possibly correlated, error source;
//! the oracle can silently condition on state the learner never sees,
//! producing a confident label for what is actually an ambiguous
//! observation; and "missing evidence" gets mapped to `0.5` by convention
//! rather than to whatever the world's actual base rate is. This module
//! exists to make an oracle's honesty checkable instead of assumed.
//!
//! [`World`] is the seam: something that can enumerate the observations it
//! can produce, together with the EXACT posterior over outcomes conditioned
//! on exactly what that observation reveals - never on anything else.
//! [`check_information_refinement`] is the load-bearing test every `World`
//! impl must pass: probability-weighted, an observation's posteriors must
//! reproduce the world's own prior exactly. An oracle that peeked at hidden
//! state to assign a confident label can fail this even when each individual
//! posterior looks plausible in isolation - it is the identity a Bayesian
//! world is required to satisfy, not a heuristic sanity check.
//!
//! [`DecisionContract`] is the explicit record of what a task actually
//! means: the event a probability is a probability OF, the loss it will be
//! judged under, and what kind of oracle produced its targets. "Will this
//! fail?" is not a well-posed question without an operating window and a
//! horizon; a contract is what makes that window explicit instead of
//! implicit in one generator's private assumptions.
//!
//! Concrete worlds (a specific device, a specific workflow) are deliberately
//! NOT built here - this crate stays domain-agnostic by design (see this
//! crate's own doc), and a one-off world belongs where it is used, the same
//! way brain's own applications keep a game or a scene generator inside the
//! sample that demonstrates it rather than inside a shared engine crate.

/// A discrete probability distribution over outcomes: `p[y] = P(Y = y)`.
pub type Distribution = Vec<f32>;

/// One observation a [`World`] can produce.
#[derive(Clone, Debug)]
pub struct Observation {
    /// A human-readable label for reporting - carries NO information a
    /// learner is meant to infer from; two observations with different names
    /// and the same evidence must carry the same [`posterior`](Self::posterior).
    pub name: String,
    /// `P(this observation)` under the world's own prior - the world decides
    /// this, not the caller.
    pub probability: f32,
    /// `P(Y | this observation)` - the oracle target for a learner that sees
    /// exactly this observation and nothing else.
    pub posterior: Distribution,
}

/// An executable probabilistic world: the source of exact oracle targets
/// [`crate::scoring`] trains against and [`crate::cost`] acts on.
///
/// Every observation this world produces must be exact WITHIN the world's
/// own stated probability model - see [`DecisionContract::oracle_kind`] for
/// how a caller distinguishes that from an estimated or empirically grounded
/// target.
pub trait World {
    /// How many values the outcome `Y` can take.
    fn n_outcomes(&self) -> usize;
    /// `P(Y)` before any observation - the belief a learner should report
    /// when no evidence is available. NOT `0.5` by convention: whatever the
    /// world's actual base rate is.
    fn prior(&self) -> Distribution;
    /// Every observation this world can produce from one query, MUTUALLY
    /// EXCLUSIVE and EXHAUSTIVE (their probabilities sum to 1) - see
    /// [`check_information_refinement`] for the identity this implies.
    fn observations(&self) -> Vec<Observation>;
}

/// What kind of oracle produced a world's posteriors, so a consumer never
/// treats an estimated or weakly-supervised target as an exact one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OracleKind {
    /// Exact within the stated probability model: a closed-form Bayesian
    /// computation, a finite probability model, a graph algorithm.
    Exact,
    /// A Monte Carlo or simulator estimate; carries a nonzero error bound.
    Estimated,
    /// Grounded in observed outcomes and a measured noise model.
    Empirical,
}

/// The explicit record of what a decision task means. Every field is
/// load-bearing: without an event definition, "will this fail" has no stated
/// answer; without a loss function, "best action" is undefined; without an
/// oracle kind, a downstream consumer cannot tell an exact target from an
/// estimated one.
#[derive(Clone, Debug)]
pub struct DecisionContract {
    pub task_name: String,
    /// What a learner is given: the schema of one [`Observation`].
    pub observation_schema: String,
    /// What the outcome distribution is OVER: the schema of `Y`.
    pub outcome_schema: String,
    /// What actions are available to condition on a [`crate::cost::CostMatrix`].
    pub action_schema: String,
    /// The event and its operating window - e.g. "the device is faulty at
    /// the time of inspection", not merely "will this fail" with an implicit
    /// and unstated horizon.
    pub event_definition: String,
    pub oracle_kind: OracleKind,
    /// Nonzero only for [`OracleKind::Estimated`]; the oracle's own claimed
    /// numerical error bound on a posterior.
    pub oracle_error_tolerance: f32,
}

/// The identity a [`World`]'s observations must satisfy: probability-weighted
/// across every observation, the posteriors must reproduce the prior
/// exactly - `prior = E_observation[posterior]`. This is the martingale
/// consistency a world's own probability model implies (nested information
/// sets: `F` = "no evidence yet", `G` = "this observation"), and it is what
/// catches an oracle that computed a posterior from hidden state the learner
/// never saw: a confident label for a genuinely ambiguous observation moves
/// the weighted average away from the prior, even though the label alone
/// looked plausible.
pub fn check_information_refinement(world: &impl World, tolerance: f32) -> Result<(), String> {
    let obs = world.observations();
    let prior = world.prior();
    let n = world.n_outcomes();
    assert_eq!(prior.len(), n, "World::prior() must have n_outcomes() entries");

    let prob_sum: f32 = obs.iter().map(|o| o.probability).sum();
    if (prob_sum - 1.0).abs() > tolerance {
        return Err(format!("observation probabilities sum to {prob_sum}, not 1 (mutual exclusivity/exhaustiveness violated)"));
    }
    for (y, &prior_y) in prior.iter().enumerate() {
        let marginal: f32 = obs.iter().map(|o| o.probability * o.posterior[y]).sum();
        if (marginal - prior_y).abs() > tolerance {
            return Err(format!("outcome {y}: E[posterior] = {marginal}, but prior = {prior_y} - an observation's posterior used information the prior did not"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::{bayes_action, CostMatrix};

    /// The device-diagnosis world from this crate's own design: a latent
    /// fault `P(F) = 0.2`, a diagnostic flag with `P(+|F) = 0.8` and
    /// `P(+|not F) = 0.1`. A private test fixture, not a published world -
    /// see this module's doc for why concrete worlds live with their
    /// consumer instead of in this crate.
    struct DiagnosisWorld;

    impl World for DiagnosisWorld {
        fn n_outcomes(&self) -> usize {
            2 // 0 = healthy, 1 = faulty
        }

        fn prior(&self) -> Distribution {
            vec![0.8, 0.2]
        }

        fn observations(&self) -> Vec<Observation> {
            let p_positive = 0.2 * 0.8 + 0.8 * 0.1; // P(+) = 0.24
            vec![
                Observation { name: "diagnostic: positive".into(), probability: p_positive, posterior: vec![1.0 / 3.0, 2.0 / 3.0] },
                Observation { name: "diagnostic: negative".into(), probability: 1.0 - p_positive, posterior: vec![18.0 / 19.0, 1.0 / 19.0] },
            ]
        }
    }

    /// An honest oracle whose posteriors merely go by a different presentation
    /// name for the identical evidence - the identity must not care.
    struct RenamedDiagnosisWorld;

    impl World for RenamedDiagnosisWorld {
        fn n_outcomes(&self) -> usize {
            2
        }
        fn prior(&self) -> Distribution {
            vec![0.8, 0.2]
        }
        fn observations(&self) -> Vec<Observation> {
            DiagnosisWorld.observations().into_iter().map(|o| Observation { name: format!("sensor reading: {}", o.name), ..o }).collect()
        }
    }

    /// A dishonest oracle: it reports the SAME posteriors as the real world
    /// but under probabilities that do not match the world's own likelihood
    /// model (as if it had peeked at the true fault state to decide how often
    /// to report each result).
    struct PeekingWorld;

    impl World for PeekingWorld {
        fn n_outcomes(&self) -> usize {
            2
        }
        fn prior(&self) -> Distribution {
            vec![0.8, 0.2]
        }
        fn observations(&self) -> Vec<Observation> {
            vec![
                Observation { name: "diagnostic: positive".into(), probability: 0.5, posterior: vec![1.0 / 3.0, 2.0 / 3.0] },
                Observation { name: "diagnostic: negative".into(), probability: 0.5, posterior: vec![18.0 / 19.0, 1.0 / 19.0] },
            ]
        }
    }

    #[test]
    fn the_honest_world_satisfies_information_refinement() {
        check_information_refinement(&DiagnosisWorld, 1e-5).expect("the worked-example world's own numbers must be self-consistent");
    }

    #[test]
    fn renaming_an_observation_does_not_change_its_posterior_or_the_identity() {
        let renamed = RenamedDiagnosisWorld.observations();
        let original = DiagnosisWorld.observations();
        for (r, o) in renamed.iter().zip(&original) {
            assert_eq!(r.posterior, o.posterior, "presentation must not leak into the target distribution");
            assert!(r.name.contains(&o.name), "the identity check does not depend on the label text");
        }
        check_information_refinement(&RenamedDiagnosisWorld, 1e-5).expect("renaming observations must not break the identity");
    }

    #[test]
    fn a_peeking_oracle_fails_the_identity_even_though_each_posterior_looks_plausible() {
        let err = check_information_refinement(&PeekingWorld, 1e-5).expect_err("mismatched observation probabilities must be caught");
        assert!(err.contains("outcome"), "the failure must name which outcome's marginal broke: {err}");
    }

    /// Ties `World` to `crate::cost`: the same world's observations, under
    /// two different cost regimes, must yield DIFFERENT Bayes actions
    /// without any observation's posterior changing at all - the cost sweep
    /// gate from `crate::cost`, now driven off a `World` impl instead of
    /// hand-typed posteriors.
    #[test]
    fn a_cost_sweep_over_a_world_changes_actions_not_posteriors() {
        let cheap_fp = CostMatrix::binary(1.0, 10.0);
        let expensive_fp = CostMatrix::binary(4.0, 10.0);
        let prior = DiagnosisWorld.prior();

        let cheap_action = bayes_action(&prior, &cheap_fp).action;
        let expensive_action = bayes_action(&prior, &expensive_fp).action;
        assert_ne!(cheap_action, expensive_action, "the boundary crossing from the worked example must reproduce here");

        // The world's own prior is a fixed value passed to both calls above -
        // nothing about DiagnosisWorld::prior() depends on the cost matrix.
        assert_eq!(DiagnosisWorld.prior(), prior);
    }
}

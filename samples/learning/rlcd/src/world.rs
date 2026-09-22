// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The device-diagnosis world: the worked example this whole sample (and
//! `brain-rlcd`'s own test suite) is built around, restated here as a real
//! `brain::World` with real rendered training text.
//!
//! A latent fault, `P(fault) = 0.2`; a diagnostic test with
//! `P(+ | fault) = 0.8` and `P(+ | healthy) = 0.1`. Three evidence states -
//! no test run yet, a positive result, a negative result - each rendered as
//! several distinct phrasings, so training sees genuine textual variety and
//! evaluation genuinely generalizes to UNSEEN phrasings of the same
//! evidence, rather than memorizing three fixed strings. Presentation is
//! deliberately decoupled from the target: every phrasing of one evidence
//! state carries the identical exact posterior.

use brain::{Distribution, Observation, RlcdExample, World};

pub const P_FAULT: f32 = 0.2;
pub const P_POSITIVE_GIVEN_FAULT: f32 = 0.8;
pub const P_POSITIVE_GIVEN_HEALTHY: f32 = 0.1;

/// `P(+)` under the world's own prior: `0.2*0.8 + 0.8*0.1 = 0.24`.
fn p_positive() -> f32 {
    P_FAULT * P_POSITIVE_GIVEN_FAULT + (1.0 - P_FAULT) * P_POSITIVE_GIVEN_HEALTHY
}

/// `P(fault | +)` by Bayes' rule: `2/3`.
fn p_fault_given_positive() -> f32 {
    (P_FAULT * P_POSITIVE_GIVEN_FAULT) / p_positive()
}

/// `P(fault | -)` by Bayes' rule: `1/19`.
fn p_fault_given_negative() -> f32 {
    (P_FAULT * (1.0 - P_POSITIVE_GIVEN_FAULT)) / (1.0 - p_positive())
}

/// Outcome names, matching `DiagnosisWorld::n_outcomes`'s index order -
/// `RlcdSpec::options` is built from this.
pub const OUTCOME_NAMES: [&str; 2] = ["healthy", "faulty"];

pub struct DiagnosisWorld;

impl World for DiagnosisWorld {
    fn n_outcomes(&self) -> usize {
        2
    }

    fn prior(&self) -> Distribution {
        vec![1.0 - P_FAULT, P_FAULT]
    }

    fn observations(&self) -> Vec<Observation> {
        let pf_pos = p_fault_given_positive();
        let pf_neg = p_fault_given_negative();
        vec![
            Observation { name: "diagnostic: positive".into(), probability: p_positive(), posterior: vec![1.0 - pf_pos, pf_pos] },
            Observation { name: "diagnostic: negative".into(), probability: 1.0 - p_positive(), posterior: vec![1.0 - pf_neg, pf_neg] },
        ]
    }
}

const NO_EVIDENCE_TEMPLATES: &[&str] = &[
    "The device has not been inspected yet.",
    "No diagnostic test has been run on this unit.",
    "We have not yet checked this device for faults.",
    "This is a new intake; no measurements have been taken.",
    "The technician has not run any diagnostics.",
    "Status: pending inspection.",
    "No sensor readings are available for this unit.",
    "The unit just arrived and has not been tested.",
];

const POSITIVE_TEMPLATES: &[&str] = &[
    "The diagnostic test came back positive.",
    "Diagnostic result: POSITIVE for a fault.",
    "The sensor flagged an anomalous reading.",
    "Test outcome: positive indication of a fault.",
    "The built-in self-test reported a failure signature.",
    "Diagnostics: FLAG RAISED.",
    "The inspection turned up a positive fault indicator.",
    "Sensor reading exceeded the fault threshold.",
];

const NEGATIVE_TEMPLATES: &[&str] = &[
    "The diagnostic test came back negative.",
    "Diagnostic result: NEGATIVE, no fault detected.",
    "The sensor reading was within normal range.",
    "Test outcome: no fault indication.",
    "The built-in self-test reported nominal operation.",
    "Diagnostics: CLEAR.",
    "The inspection found no fault indicators.",
    "Sensor reading stayed below the fault threshold.",
];

/// How many of each evidence class's templates are held out for evaluation -
/// the LAST `EVAL_HOLDOUT` phrasings of each class, never seen in training,
/// so held-out ECE and regret measure generalization to unseen wording
/// rather than memorization of the training strings.
const EVAL_HOLDOUT: usize = 2;

fn examples_for(templates: &[&str], target: &[f32]) -> (Vec<RlcdExample>, Vec<RlcdExample>) {
    let split = templates.len() - EVAL_HOLDOUT.min(templates.len());
    let train = templates[..split].iter().map(|&t| RlcdExample::new(t, target.to_vec())).collect();
    let eval = templates[split..].iter().map(|&t| RlcdExample::new(t, target.to_vec())).collect();
    (train, eval)
}

/// Every template phrasing of every evidence state, paired with that state's
/// EXACT oracle posterior: `(train, eval)`.
pub fn examples() -> (Vec<RlcdExample>, Vec<RlcdExample>) {
    let world = DiagnosisWorld;
    let obs = world.observations();

    let (mut train, mut eval) = examples_for(NO_EVIDENCE_TEMPLATES, &world.prior());
    for (templates, o) in [(POSITIVE_TEMPLATES, &obs[0]), (NEGATIVE_TEMPLATES, &obs[1])] {
        let (t, e) = examples_for(templates, &o.posterior);
        train.extend(t);
        eval.extend(e);
    }
    (train, eval)
}

/// Every phrasing of the "no evidence yet" class - what
/// `RlcdPipeline::train_voi_policy` trains and evaluates the meta-decision
/// (block/release/inspect) on, since that decision is only ever made before
/// any query result is in hand.
pub fn no_evidence_states() -> Vec<String> {
    NO_EVIDENCE_TEMPLATES.iter().map(|&s| s.to_string()).collect()
}

/// The canonical (first) phrasing of an evidence class, keyed by the SAME
/// name `witness::search` labels its candidate points with. An
/// [`Observation`] carries no text of its own (only its exact posterior), so
/// probing a real trained model for witnesses needs this: a fixed mapping
/// back to one representative rendering per evidence class - see `main.rs`.
pub fn canonical_render(observation_name: &str) -> &'static str {
    match observation_name {
        "no evidence yet" => NO_EVIDENCE_TEMPLATES[0],
        "diagnostic: positive" => POSITIVE_TEMPLATES[0],
        "diagnostic: negative" => NEGATIVE_TEMPLATES[0],
        other => panic!("world.rs: no canonical rendering for observation {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain::check_information_refinement;

    #[test]
    fn the_worlds_own_numbers_match_the_worked_example() {
        assert!((p_positive() - 0.24).abs() <= 1e-6);
        assert!((p_fault_given_positive() - 2.0 / 3.0).abs() <= 1e-6);
        assert!((p_fault_given_negative() - 1.0 / 19.0).abs() <= 1e-6);
    }

    #[test]
    fn the_world_satisfies_information_refinement() {
        check_information_refinement(&DiagnosisWorld, 1e-5).expect("this world's own numbers must be self-consistent");
    }

    #[test]
    fn every_template_class_splits_into_a_nonempty_train_and_eval_set() {
        let (train, eval) = examples();
        assert_eq!(train.len(), 18, "3 classes * (8 - 2 held out) templates");
        assert_eq!(eval.len(), 6, "3 classes * 2 held out");
        for ex in train.iter().chain(&eval) {
            assert_eq!(ex.target.len(), 2);
            let sum: f32 = ex.target.iter().sum();
            assert!((sum - 1.0).abs() <= 1e-5, "target must be a distribution: {:?}", ex.target);
        }
    }

    #[test]
    fn canonical_render_covers_every_witness_candidate_label() {
        canonical_render("no evidence yet");
        canonical_render("diagnostic: positive");
        canonical_render("diagnostic: negative");
    }
}

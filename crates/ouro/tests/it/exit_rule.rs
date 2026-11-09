// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The exit rule picks the FIRST pass whose cumulative exit mass reaches the
//! threshold. The cumulative mass only grows, so a rule without that "first"
//! returns the last pass for every threshold; every other test runs at
//! threshold 1.0, where the two coincide, so this one uses thresholds below it.

use ouro::model::exit_pass;

/// The gate logit whose sigmoid is `p`.
fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

#[test]
fn the_first_pass_reaching_the_threshold_is_chosen() {
    // Exit probabilities 0.5 at pass 0, then 0.5 of the remainder: the
    // cumulative mass is 0.5, 0.75, 0.875, 1.0 over four passes.
    let gates = [logit(0.5), logit(0.5), logit(0.5), 0.0];
    assert_eq!(exit_pass(&gates, 0.5), 0);
    assert_eq!(exit_pass(&gates, 0.6), 1);
    assert_eq!(exit_pass(&gates, 0.8), 2);
    assert_eq!(exit_pass(&gates, 0.9), 3);
    assert_eq!(exit_pass(&gates, 1.0), 3, "the last pass takes all remaining mass");
}

#[test]
fn the_last_gate_is_ignored_because_the_last_pass_takes_the_rest() {
    let a = [logit(0.1), logit(0.1), -30.0];
    let b = [logit(0.1), logit(0.1), 30.0];
    for th in [0.05, 0.15, 0.5, 1.0] {
        assert_eq!(exit_pass(&a, th), exit_pass(&b, th), "threshold {th}");
    }
}

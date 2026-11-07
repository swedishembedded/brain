// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gradient-check gate for the Ripple training graph, on whichever backend `BRAIN_DEVICE` selects.
//!
//! Finite differences of the same forward are the only evidence that the backward differentiates the
//! value-residual lerp (into `x0`, the normed stream and the per-layer gate), the f32 sigmoid gate, both
//! clamps in both regimes, the per-head QK norms, the windowed and global attention and the embedding scale.

use gradcheck::ricursive::{check_ricursive, check_ricursive_folded};
use gradcheck::Report;

const ATOL: f32 = 4e-3;
const RTOL: f32 = 8.0e-2;

fn gate(report: Report, what: &str) {
    report.print();
    let fails = report.failures(ATOL, RTOL);
    assert!(fails.is_empty(), "{what}: gradient check failed for {:?}", fails.iter().map(|c| (&c.param, c.analytic, c.numeric, c.rel_err)).collect::<Vec<_>>());
}

#[test]
fn ricursive_every_parameter_matches_finite_differences() {
    for seed in [3u64, 11] {
        gate(check_ricursive(seed), &format!("tiny ricursive, seed {seed}"));
    }
}

#[test]
fn ricursive_folded_tensors_match_entry_by_entry() {
    for seed in [3u64, 11] {
        gate(check_ricursive_folded(seed), &format!("tiny ricursive folded, seed {seed}"));
    }
}

/// The checks above prove nothing about a path the batch never exercises: switching each feature off must move the
/// loss, so the window excludes keys, both clamps fire, and the gate, the value residual and the scale are live.
#[test]
fn ricursive_every_feature_changes_the_loss() {
    for (feature, moved) in gradcheck::ricursive::feature_effects(3) {
        assert!(moved > 1e-3, "{feature} changes the loss by only {moved}: the gradient check would not see it");
    }
}

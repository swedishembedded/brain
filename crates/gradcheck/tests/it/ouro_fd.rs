// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gradient-check gate for Ouro's looped training graph, on whichever backend
//! `BRAIN_DEVICE` selects (run it under both `cpu` and the GPU).
//!
//! The shared stack is visited `total_ut_steps` times per token and every
//! pass's loss is weighted by its exit probability; finite differences of the
//! same forward are the only evidence that the one parameter set accumulated
//! all of those uses.

use gradcheck::{check_ouro, check_ouro_exit_entropy, Report};

const ATOL: f32 = 4e-3;
const RTOL: f32 = 8.0e-2;

fn gate(report: Report, what: &str) {
    report.print();
    let fails = report.failures(ATOL, RTOL);
    assert!(
        fails.is_empty(),
        "{what}: gradient check failed for {:?}",
        fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>()
    );
    let dead = report.dead_gradients();
    assert!(
        dead.is_empty(),
        "{what}: silently-dead gradients {:?}",
        dead.iter().map(|c| (&c.param, c.analytic, c.numeric)).collect::<Vec<_>>()
    );
}

#[test]
fn ouro_shared_stack_accumulates_every_pass() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    for seed in [3u64, 11] {
        gate(check_ouro(seed), &format!("tiny ouro, seed {seed}"));
    }
}

/// The entropy-regularised objective (the exit distribution's entropy weighed in on the supervised rows): the gate
/// gradient carries the entropy term's derivative, at the reference's weights and a strong one.
#[test]
fn ouro_exit_entropy_reaches_the_gates() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    for (seed, beta) in [(3u64, 0.1f32), (11, 0.05), (5, 1.0)] {
        gate(check_ouro_exit_entropy(seed, beta), &format!("tiny ouro, seed {seed}, exit entropy {beta}"));
    }
}

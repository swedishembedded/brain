// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gradient-check gate for Ouro's looped training graph, on whichever backend
//! `BRAIN_DEVICE` selects (run it under both `cpu` and the GPU).
//!
//! The shared stack is visited `total_ut_steps` times per token and every
//! pass's loss is weighted by its exit probability; finite differences of the
//! same forward are the only evidence that the one parameter set accumulated
//! all of those uses.

use gradcheck::{check_ouro, Report};

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

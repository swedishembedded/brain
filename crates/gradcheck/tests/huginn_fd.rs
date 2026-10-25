// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gradient-check gate for Huginn's training graph, on whichever backend the
//! run selects (run it under both the host tier and a GPU).
//!
//! Every recurrent step applies the same four core blocks, so the shared
//! parameters' analytic gradient has to be the sum over those applications.

use gradcheck::{check_huginn, Report};

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
fn huginn_shared_core_accumulates_every_step() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    for seed in [7u64, 19] {
        gate(check_huginn(seed), &format!("tiny huginn, seed {seed}"));
    }
}

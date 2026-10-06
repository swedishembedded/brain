// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gradient-check gate for `horizon`'s training graph, on whichever backend
//! `BRAIN_DEVICE` selects (run it under both `cpu` and the GPU).

use gradcheck::horizon::{
    check_horizon, check_horizon_additive, check_horizon_attention, check_horizon_next_events,
    check_horizon_visits,
};
use gradcheck::Report;

const ATOL: f32 = 4e-3;
const RTOL: f32 = 8e-2;

fn gate(report: Report, what: &str) {
    report.print();
    let fails = report.failures(ATOL, RTOL);
    assert!(
        fails.is_empty(),
        "{what}: gradient check failed for {:?}",
        fails
            .iter()
            .map(|c| (&c.param, c.abs_err, c.rel_err))
            .collect::<Vec<_>>()
    );
    let dead = report.dead_gradients();
    assert!(
        dead.is_empty(),
        "{what}: silently-dead gradients {:?}",
        dead.iter()
            .map(|c| (&c.param, c.analytic, c.numeric))
            .collect::<Vec<_>>()
    );
}

#[test]
fn horizon_analytic_grads_match_finite_differences() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    for seed in [3u64, 11] {
        gate(check_horizon(seed), &format!("set encoder, seed {seed}"));
    }
}

#[test]
fn horizon_next_event_group_analytic_grads_match_finite_differences() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    for seed in [3u64, 11] {
        gate(
            check_horizon_next_events(seed),
            &format!("next-event group, seed {seed}"),
        );
    }
}

#[test]
fn horizon_additive_analytic_grads_match_finite_differences() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    for seed in [3u64, 11] {
        gate(
            check_horizon_additive(seed),
            &format!("additive, seed {seed}"),
        );
    }
}

#[test]
fn horizon_visits_analytic_grads_match_finite_differences() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    for seed in [3u64, 11] {
        gate(
            check_horizon_visits(seed),
            &format!("state across visits, seed {seed}"),
        );
    }
}

#[test]
fn horizon_attention_analytic_grads_match_finite_differences() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    for seed in [3u64, 11] {
        gate(
            check_horizon_attention(seed),
            &format!("attention across visits, seed {seed}"),
        );
    }
}

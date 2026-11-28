// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gradient-check gate for `horizon`'s training graph, on the device the test
//! harness names (`BRAIN_TEST_DEVICE`; run it under both `cpu` and a GPU).

use gradcheck::horizon::{
    check_horizon, check_horizon_additive, check_horizon_attention, check_horizon_gdn,
    check_horizon_hybrid, check_horizon_next_events, check_horizon_visits,
};
use gradcheck::Report;

/// The harness's device, named to the model.
fn placement() -> horizon::placement::Placement {
    horizon::placement::Placement::Device(brain_testutil::device::device().device.clone())
}

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
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    for seed in [3u64, 11] {
        gate(check_horizon(seed, &placement()), &format!("set encoder, seed {seed}"));
    }
}

#[test]
fn horizon_next_event_group_analytic_grads_match_finite_differences() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    for seed in [3u64, 11] {
        gate(
            check_horizon_next_events(seed, &placement()),
            &format!("next-event group, seed {seed}"),
        );
    }
}

#[test]
fn horizon_additive_analytic_grads_match_finite_differences() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    for seed in [3u64, 11] {
        gate(
            check_horizon_additive(seed, &placement()),
            &format!("additive, seed {seed}"),
        );
    }
}

#[test]
fn horizon_visits_analytic_grads_match_finite_differences() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    for seed in [3u64, 11] {
        gate(
            check_horizon_visits(seed, &placement()),
            &format!("state across visits, seed {seed}"),
        );
    }
}

#[test]
fn horizon_attention_analytic_grads_match_finite_differences() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    for seed in [3u64, 11] {
        gate(
            check_horizon_attention(seed, &placement()),
            &format!("attention across visits, seed {seed}"),
        );
    }
}

#[test]
fn horizon_gated_deltanet_analytic_grads_match_finite_differences() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    for seed in [3u64, 11] {
        gate(
            check_horizon_gdn(seed, &placement()),
            &format!("gated deltanet stack, seed {seed}"),
        );
    }
}

#[test]
fn horizon_hybrid_analytic_grads_match_finite_differences() {
    if brain_testutil::gpu_tests_skipped() {
        return;
    }
    for seed in [3u64, 11] {
        gate(
            check_horizon_hybrid(seed, &placement()),
            &format!("3:1 hybrid stack, seed {seed}"),
        );
    }
}

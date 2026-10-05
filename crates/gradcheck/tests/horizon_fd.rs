// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gradient-check gate for `horizon`'s training graph, on whichever backend
//! `BRAIN_DEVICE` selects (run it under both `cpu` and the GPU).

use gradcheck::horizon::check_horizon;

const ATOL: f32 = 4e-3;
const RTOL: f32 = 8e-2;

#[test]
fn horizon_analytic_grads_match_finite_differences() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    for seed in [3u64, 11] {
        let report = check_horizon(seed);
        report.print();
        let fails = report.failures(ATOL, RTOL);
        assert!(
            fails.is_empty(),
            "seed {seed}: gradient check failed for {:?}",
            fails
                .iter()
                .map(|c| (&c.param, c.abs_err, c.rel_err))
                .collect::<Vec<_>>()
        );
        let dead = report.dead_gradients();
        assert!(
            dead.is_empty(),
            "seed {seed}: silently-dead gradients {:?}",
            dead.iter()
                .map(|c| (&c.param, c.analytic, c.numeric))
                .collect::<Vec<_>>()
        );
    }
}

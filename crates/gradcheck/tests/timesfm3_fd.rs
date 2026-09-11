// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gradient-check gate for **TimesFM-3**'s training graph.
//!
//! `crates/gradcheck/src/timesfm3.rs`'s `check_*` entries are library
//! functions; these tests are what actually runs them, on whichever backend
//! `BRAIN_DEVICE` selects. An entry point not wired into a test here is not a
//! gate.
//!
//! Every test is gated on `MOE_SKIP_GPU_TESTS` like every other GPU-touching
//! test, and reports through `Report::print` so a failure names the offending
//! tensor.

use gradcheck::timesfm3 as tfm;
use gradcheck::Report;

/// fp32 directional FD on a device: the workspace-standard combined
/// tolerance, never loosened.
const ATOL: f32 = 4e-3;
const RTOL: f32 = 8e-2;

fn skip_gpu() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn gate(report: Report, what: &str) {
    report.print();
    let fails = report.failures(ATOL, RTOL);
    assert!(
        fails.is_empty(),
        "{what} gradient check failed for {:?}",
        fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>()
    );
    // A gradient that came back identically zero against a clearly non-zero
    // numeric one passes `within`'s absolute floor - the shape a missing or
    // wrongly-routed kernel dispatch presents as. This graph has two tensor
    // families with no kernel of their own (both halves of the PerDimScale
    // fold, produced by a host-side split), so the guard is not theoretical.
    let dead = report.dead_gradients();
    assert!(dead.is_empty(), "{what}: dead (identically zero) gradients for {:?}", dead.iter().map(|c| &c.param).collect::<Vec<_>>());
}

#[test]
fn timesfm3_analytic_grads_match_finite_differences() {
    if skip_gpu() {
        brain_testutil::skip_unavailable("check_timesfm3: MOE_SKIP_GPU_TESTS set");
        return;
    }
    gate(tfm::check_timesfm3(7), "TimesFM-3 core");
}

#[test]
fn timesfm3_one_layer_analytic_grads_match_finite_differences() {
    if skip_gpu() {
        brain_testutil::skip_unavailable("check_timesfm3_one_layer: MOE_SKIP_GPU_TESTS set");
        return;
    }
    gate(tfm::check_timesfm3_one_layer(7), "TimesFM-3 core (single layer)");
}

/// The PerDimScale fold, per entry. `check_timesfm3` reaches this tensor only
/// through a contraction that best-of-4 selects to MINIMISE, which AGENTS.md
/// records as blind to a partial gradient error - and a host-side split of
/// one `d(effective gain)` into two originals is exactly where a partial
/// error lives. Both halves, because they enter as a product.
#[test]
fn timesfm3_per_dim_scale_fold_splits_correctly_per_entry() {
    if skip_gpu() {
        brain_testutil::skip_unavailable("check_timesfm3_per_dim_scale_elementwise: MOE_SKIP_GPU_TESTS set");
        return;
    }
    gate(tfm::check_timesfm3_per_dim_scale_elementwise(7), "TimesFM-3 per_dim_scale (per entry)");
    gate(tfm::check_timesfm3_query_ln_elementwise(7), "TimesFM-3 query_ln (per entry)");
}

/// The eps probe, run as a gate rather than left as a comment: it asserts
/// that the chosen `5e-4` is not sitting on a knee, i.e. that the max
/// relative error there is no worse than at the decade on either side. If a
/// future change to the graph moves the knee, this fails and prints the
/// table, which is the reporting AGENTS.md asks for instead of a widened
/// bound.
#[test]
fn timesfm3_eps_plateau() {
    if skip_gpu() {
        brain_testutil::skip_unavailable("check_timesfm3_eps_sweep: MOE_SKIP_GPU_TESTS set");
        return;
    }
    let table = tfm::check_timesfm3_eps_sweep(7);
    for (eps, rel) in &table {
        println!("  eps={eps:.1e}  max_rel={rel:.3e}");
    }
    let at = |e: f32| table.iter().find(|(x, _)| *x == e).expect("eps in table").1;
    assert!(at(5e-4) <= RTOL, "eps 5e-4 max_rel {:.3e} exceeds rtol", at(5e-4));
    assert!(at(5e-4) <= at(5e-3).max(at(5e-5)) * 4.0, "eps 5e-4 is not on the plateau: {table:?}");
}

/// The eps probe for the per-entry fold check, gated the same way.
#[test]
fn timesfm3_per_dim_scale_eps_plateau() {
    if skip_gpu() {
        brain_testutil::skip_unavailable("check_timesfm3_per_dim_scale_eps_sweep: MOE_SKIP_GPU_TESTS set");
        return;
    }
    let table = tfm::check_timesfm3_per_dim_scale_eps_sweep(7);
    for (eps, rel) in &table {
        println!("  eps={eps:.1e}  max_rel={rel:.3e}");
    }
    let at = |e: f32| table.iter().find(|(x, _)| *x == e).expect("eps in table").1;
    assert!(at(1e-2) <= RTOL, "eps 1e-2 max_rel {:.3e} exceeds rtol", at(1e-2));
    assert!(at(1e-2) <= at(2e-2).max(at(1e-3)) * 4.0, "eps 1e-2 is not on the plateau: {table:?}");
}

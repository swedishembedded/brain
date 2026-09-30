// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gradient-check gate for the SAM tower's DeepSeek-VL HD branch and neck
//! resize (`gradcheck::sam_hd`).

use gradcheck::Report;

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
        "{what}: gradient check failed for {:?}",
        fails.iter().map(|c| (&c.param, c.abs_err, c.rel_err)).collect::<Vec<_>>()
    );
    let dead = report.dead_gradients();
    assert!(dead.is_empty(), "{what}: dead (identically zero) gradients for {:?}", dead.iter().map(|c| &c.param).collect::<Vec<_>>());
    println!("{what}: {} checks, worst rel {:.3e}", report.checks.len(), report.max_rel());
}

/// Every tensor of the HD tower: both necks, the blocks the HD adjoint joins
/// into, the resize adjoint on both paths.
#[test]
fn sam_hd_analytic_grads_match_finite_differences() {
    if skip_gpu() {
        brain_testutil::skip_unavailable("check_sam_hd: MOE_SKIP_GPU_TESTS set");
        return;
    }
    gate(gradcheck::check_sam_hd(7), "SAM HD tower");
}

/// `hd_alpha` and the compressor weights both paths share, per entry: the
/// shared weights' gradient must be the sum over both uses.
#[test]
fn sam_hd_shared_compressor_and_alpha_per_entry() {
    if skip_gpu() {
        brain_testutil::skip_unavailable("check_sam_hd_shared: MOE_SKIP_GPU_TESTS set");
        return;
    }
    gate(gradcheck::check_sam_hd_shared(7), "SAM HD shared compressor + hd_alpha (per entry)");
}

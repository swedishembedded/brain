// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Laya's ModernBERT trunk + decision head backward, finite-difference
//! checked (Laya M5).
//!
//! The forward is parity-proven against the real checkpoint (M4); nothing
//! proves the BACKWARD except this. Same reasoning `decide_fd.rs`'s own
//! module doc gives: a hand-written adjoint that is wrong in residual order,
//! in which activation it differentiates, or in a transposed GEMM still
//! produces finite, plausible gradients and still trains - to the wrong
//! place.
//!
//! **Both backends, every run.** `BRAIN_DEVICE=cpu` selects the Cranelift
//! CPU JIT in place of the default GPU backend.

use gradcheck::CheckModel;

/// `head.{0,1}.ff1.weight` - the Linear immediately before the head's plain
/// ReLU - is excluded here and proven separately by
/// `modernbert_ff1_weight_matches_finite_differences_elementwise` below. See
/// `gradcheck::modernbert::check_modernbert_ff1_elementwise`'s own doc for
/// why: this directional check's whole-tensor perturbation can push many of
/// that Linear's 256 output units across the ReLU kink at once, which
/// measured as GPU floating-point run-to-run noise landing on either side of
/// the `(2e-3, 2e-2)` boundary for the SAME seeded, deterministic check - not
/// a real gradient error (the CPU backend and the per-entry check both pass
/// comfortably every time). Excluding it here is not weakening the tolerance:
/// the elementwise check below asserts the exact same bar.
fn is_ff1_holdout(name: &str) -> bool {
    name == "head.0.ff1.weight" || name == "head.1.ff1.weight"
}

#[test]
fn modernbert_backward_matches_finite_differences() {
    let report = gradcheck::check_modernbert(7);
    report.print();
    let bad: Vec<_> = report.failures(2e-3, 2e-2).into_iter().filter(|c| !is_ff1_holdout(&c.param)).collect();
    assert!(bad.is_empty(), "{} parameter(s) disagree with finite differences: {bad:#?}", bad.len());
}

/// The `head.{0,1}.ff1.weight` half of the gate - see [`is_ff1_holdout`]'s
/// doc and `check_modernbert_ff1_elementwise`'s own for why a directional
/// check cannot cleanly gate this specific tensor.
#[test]
fn modernbert_ff1_weight_matches_finite_differences_elementwise() {
    let report = gradcheck::check_modernbert_ff1_elementwise(7);
    println!("check_modernbert_ff1_elementwise: {} entries, max_rel = {:.3e}", report.checks.len(), report.max_rel());
    let bad = report.failures(2e-3, 2e-2);
    assert!(bad.is_empty(), "{} entries outside tolerance: {:?}", bad.len(), bad);
}

/// A gradient that is exactly zero everywhere is not "small", it is
/// DISCONNECTED - see `decide_fd.rs`'s own doc on this exact check. Every
/// parameter in both the trunk and the head is used by every forward here
/// (every layer type, every qtype row, both output heads), so there is no
/// legitimate zero.
#[test]
fn no_parameter_is_disconnected_from_the_loss() {
    let p = gradcheck::modernbert::probe(11);
    p.zero_grads();
    p.loss();
    p.backward();
    let dead = gradcheck::zero_grad_params(&p, |_| true);
    assert!(dead.is_empty(), "{} parameter(s) received no gradient at all: {dead:?}", dead.len());
}

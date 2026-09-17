// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The decision encoder's backward, finite-difference checked.
//!
//! The forward is parity-proven against the released checkpoint; nothing proves
//! the BACKWARD except this. A hand-written adjoint that is wrong in the
//! residual order, in which activation it differentiates, or in a transposed
//! GEMM still produces finite, plausible gradients and still trains - to the
//! wrong place.
//!
//! **Both backends, every run.** This repository has already paid for a
//! workgroup-barrier reduction that returned all-zero gradients on the CPU
//! backend alone; a check that only ever ran on the GPU called it green.
//! `BRAIN_DEVICE=cpu` selects the other one.

use gradcheck::CheckModel;

#[test]
fn encoder_backward_matches_finite_differences() {
    let report = gradcheck::check_decide(7);
    report.print();
    let bad = report.failures(2e-3, 2e-2);
    assert!(bad.is_empty(), "{} parameter(s) disagree with finite differences: {bad:#?}", bad.len());
}

/// A gradient that is exactly zero everywhere is not "small", it is
/// DISCONNECTED: some path in the reverse pass never reached that parameter.
/// Every parameter here is used by every forward, so there is no legitimate
/// zero - and a directional check averages over a whole tensor, which is
/// exactly where one dead parameter hides inside a tensor that is otherwise
/// right.
#[test]
fn no_parameter_is_disconnected_from_the_loss() {
    let p = gradcheck::decide::probe(11);
    p.zero_grads();
    p.loss();
    p.backward();
    let dead = gradcheck::zero_grad_params(&p, |_| true);
    assert!(dead.is_empty(), "{} parameter(s) received no gradient at all: {dead:?}", dead.len());
}

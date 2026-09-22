// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Laya's decision head has a gradient-checked backward (`modernbert_fd.rs`,
//! Laya M5) and, as of Laya M6, an AdamW step
//! (`modernbert::laya::LayaHead::adamw_step`/`adamw_step_scaled`) built on
//! the same `ParamStore` + `optim::Optim` primitive `decide::model::Encoder`
//! uses. AdamW itself is a well-tested, model-agnostic primitive over
//! `(param, grad)` pairs - it needs no NEW numerical derivation to verify,
//! only confirmation that THIS wiring calls it correctly: the right
//! `ParamStore`, the right device handle, a real (not stale/zero) gradient.
//! That is what "loss decreases on a fixed micro-run" catches here - a
//! wrong-store or wrong-handle step leaves the loss flat or unchanged; a
//! sign error makes it increase.
//!
//! **The loss used is deliberately unbounded, so it diverges rather than
//! converges to a floor - that is expected, not a bug.** It reuses
//! `gradcheck::modernbert::Probe`'s own fixed-weight linear functional of
//! the two output heads (the same objective the finite-difference check in
//! `modernbert_fd.rs` verifies gradients against), which exists to excite
//! every parameter's derivative for an FD check, not to have a minimum.
//! Genuine convergence toward a bounded target with this exact
//! `optim::Optim` primitive is already established empirically -
//! `samples/learning/rlcd`'s measured run trains a `Decide`-backed model
//! (the SAME optimizer primitive, a different `ParamStore`) from loss 0.64
//! to 0.21 against a real cross-entropy target.

use gradcheck::CheckModel;

#[test]
fn laya_head_training_loss_decreases_on_a_fixed_micro_run() {
    let mut p = gradcheck::modernbert::probe(19);

    let first = {
        p.zero_grads();
        let l = p.loss();
        println!("step  0  loss {l:.6}");
        p.backward();
        p.adamw_step_head(3e-2, 0.01, Some(1.0));
        l
    };

    let mut last = first;
    for step in 1..30 {
        p.zero_grads();
        last = p.loss();
        println!("step {step:>2}  loss {last:.6}");
        p.backward();
        p.adamw_step_head(3e-2, 0.01, Some(1.0));
    }

    assert!(last.is_finite(), "loss went non-finite over 30 steps: {last}");
    assert!(last < first, "loss did not decrease: first {first:.6}, last (step 30) {last:.6}");
}

/// The same run, twice, must produce the identical loss trajectory - an
/// AdamW step is: read a param + its moments, write both back. A step whose
/// device dispatches race (the exact defect `crates/decide`'s own encoder/
/// head adamw split guards against - see `LayaHead::adamw_step_scaled`'s own
/// doc) would show up as a run that is not reproducible from the same seed,
/// not necessarily as a crash.
#[test]
fn laya_head_training_is_deterministic_for_a_fixed_seed() {
    fn run() -> Vec<f32> {
        let mut p = gradcheck::modernbert::probe(23);
        let mut losses = Vec::new();
        for _ in 0..10 {
            p.zero_grads();
            losses.push(p.loss());
            p.backward();
            p.adamw_step_head(3e-2, 0.01, Some(1.0));
        }
        losses
    }
    assert_eq!(run(), run());
}

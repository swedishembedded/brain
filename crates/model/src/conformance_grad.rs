// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gradient checks of the conformance suite ([`crate::conformance`]): the
//! weighted loss against plain training (C2), against finite differences (C3)
//! and the frozen-base tiers against the f32 base (C9).
//!
//! The finite differences here are a few lines of their own rather than
//! `gradcheck::directional_check`: `gradcheck` depends on this crate, so a
//! library module here cannot depend on it.
//!
//! Swedish Embedded AB implements solutions for verifying that every model in
//! a training stack honours the same fine-tuning contract, for its clients. If
//! your team needs expertise in gradient verification for ML training code you
//! can procure our services by sending an email to info@swedishembedded.com.

use data::rng::Rng;

use crate::adapter::device::is_adapter_param;
use crate::conformance::{batch, build, trained_rank, Subject, BATCH};
use crate::training::require_token_objectives;
use crate::{Batch, Model, ModelConfig};

/// Step of the central differences, and the `allclose` bound
/// `|analytic - numeric| <= ATOL + RTOL * max(|analytic|, |numeric|)`.
const EPS: f32 = 5e-3;
const ATOL: f32 = 4e-3;
const RTOL: f32 = 8e-2;
/// A gradient tensor smaller than this fraction of the largest one has no
/// signal for the tier comparison (see [`tiers_track_the_f32_gradients`]).
const SIGNAL_FLOOR: f32 = 1e-3;

/// Give every adapter tensor small random values, so a zero-initialised `B`
/// does not make every `A` gradient vanish. A model without adapters is left alone.
pub fn randomize_adapters<M: Model>(m: &M, seed: u64) {
    let mut rng = Rng::new(seed);
    for name in m.param_names().into_iter().filter(|n| is_adapter_param(n)) {
        let n = m.read_weight(&name).len();
        let w: Vec<f32> = (0..n).map(|_| 0.2 * (rng.next_f32() - 0.5)).collect();
        m.write_weight(&name, &w);
    }
}

/// The gradients of the trainable tensors for the batch already set.
fn gradients<M: Model>(m: &M) -> Vec<(String, Vec<f32>)> {
    m.zero_grads();
    m.forward();
    m.backward();
    m.poll_wait();
    m.param_names().into_iter().map(|n| (n.clone(), m.read_grad(&n))).collect()
}

fn max_abs(v: &[f32]) -> f32 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// C2: an all-ones [`Batch::LmWeighted`] on a weighted-loss model has the loss
/// and gradients of plain [`Batch::Lm`] training.
pub fn unit_weights_reproduce_plain_training<M: Model>(subject: &Subject<M>, seed: u64) -> Result<(), String> {
    let rank = trained_rank(subject);
    let (plain, _) = build(subject, rank, BATCH, seed);
    let (mut weighted, _) = build(subject, rank, BATCH, seed);
    randomize_adapters(&plain, seed);
    randomize_adapters(&weighted, seed);
    let (tokens, targets) = batch(subject_vocab(subject));
    plain.set_batch(Batch::Lm { tokens: &tokens, targets: &targets });
    let want_grads = gradients(&plain);
    let want_loss = plain.forward();

    require_token_objectives(&mut weighted).map_err(|e| e.to_string())?.enable_weighted_loss();
    let ones = vec![1.0f32; tokens.len()];
    weighted.set_batch(Batch::LmWeighted { tokens: &tokens, targets: &targets, weights: &ones });
    let got_grads = gradients(&weighted);
    let got_loss = weighted.forward();

    if (got_loss - want_loss).abs() > 1e-5 * want_loss.abs().max(1.0) {
        return Err(format!("{}: all-ones weighted loss {got_loss} vs plain {want_loss}", subject.name));
    }
    for ((name, got), (_, want)) in got_grads.iter().zip(&want_grads) {
        let diff = got.iter().zip(want).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        if diff > 1e-5 * max_abs(want).max(1.0) {
            return Err(format!("{}: {name}: all-ones weighted gradient differs from plain by {diff}", subject.name));
        }
    }
    Ok(())
}

pub(crate) fn subject_vocab<M: Model>(subject: &Subject<M>) -> u32 {
    ModelConfig::vocab(&(subject.config)(None))
}

/// C3: the gradient of the weighted loss (non-uniform weights including exact
/// zeros), at the unit logit scale and at `1/2`, against central finite
/// differences along random directions of each trainable tensor.
pub fn weighted_loss_gradient_matches_finite_differences<M: Model>(subject: &Subject<M>, seed: u64) -> Result<(), String> {
    let (mut m, _) = build(subject, trained_rank(subject), BATCH, seed);
    randomize_adapters(&m, seed);
    require_token_objectives(&mut m).map_err(|e| e.to_string())?.enable_weighted_loss();
    let (tokens, targets) = batch(subject_vocab(subject));
    let weights: Vec<f32> = (0..tokens.len()).map(|i| if i % 3 == 0 { 0.0 } else { 0.5 + 0.1 * i as f32 }).collect();
    let mut rng = Rng::new(seed ^ 0x5a17);
    for scale in [1.0f32, 0.5] {
        require_token_objectives(&mut m).map_err(|e| e.to_string())?.set_logit_scale(scale);
        m.set_batch(Batch::LmWeighted { tokens: &tokens, targets: &targets, weights: &weights });
        let analytic = gradients(&m);
        for index in subject.budget.pick(analytic.len()) {
            let (name, g) = &analytic[index];
            let w0 = m.read_weight(name);
            let mut best: Option<(f32, f32, f32)> = None;
            for _ in 0..subject.budget.directions.max(1) {
                let v: Vec<f32> = (0..w0.len()).map(|_| if rng.next_f32() < 0.5 { -1.0 } else { 1.0 }).collect();
                let a: f32 = g.iter().zip(&v).map(|(g, v)| g * v).sum();
                let shifted = |sign: f32| -> f32 {
                    let w: Vec<f32> = w0.iter().zip(&v).map(|(w, v)| w + sign * EPS * v).collect();
                    m.write_weight(name, &w);
                    m.forward()
                };
                let numeric = (shifted(1.0) - shifted(-1.0)) / (2.0 * EPS);
                m.write_weight(name, &w0);
                let err = (a - numeric).abs() - (ATOL + RTOL * a.abs().max(numeric.abs()));
                if best.is_none_or(|(e, _, _)| err < e) {
                    best = Some((err, a, numeric));
                }
            }
            let (err, a, numeric) = best.expect("at least one direction");
            if err > 0.0 {
                return Err(format!("{}: {name} at logit scale {scale}: analytic {a} vs finite difference {numeric}", subject.name));
            }
        }
    }
    Ok(())
}

/// C9: for each declared frozen-base tier, the gradients of the trainable
/// tensors are within the tier's stated tolerance of the f32 base's.
pub fn tiers_track_the_f32_gradients<M: Model>(subject: &Subject<M>, seed: u64) -> Result<(), String> {
    let rank = trained_rank(subject);
    let (tokens, targets) = batch(subject_vocab(subject));
    let (reference, init) = build(subject, rank, BATCH, seed);
    randomize_adapters(&reference, seed);
    reference.set_batch(Batch::Lm { tokens: &tokens, targets: &targets });
    let want = gradients(&reference);
    for tier in &subject.tiers {
        let m = (tier.build)((subject.config)(rank), BATCH, crate::conformance::SEQ, &init);
        randomize_adapters(&m, seed);
        m.set_batch(Batch::Lm { tokens: &tokens, targets: &targets });
        let got = gradients(&m);
        // A tensor whose gradient is a rounding-level fraction of the largest
        // one (a gated path that is nearly closed at a random initialisation)
        // carries no signal to compare: its relative error is noise.
        let floor = SIGNAL_FLOOR * want.iter().map(|(_, w)| norm(w)).fold(0.0, f32::max);
        for ((name, g), (_, w)) in got.iter().zip(&want) {
            let diff: Vec<f32> = g.iter().zip(w).map(|(a, b)| a - b).collect();
            let rel = norm(&diff) / norm(w).max(floor).max(1e-12);
            if rel > tier.grad_rel_tol {
                return Err(format!("{}: tier {}: {name}: gradient is {rel} from the f32 base's, over the allowed {}", subject.name, tier.name, tier.grad_rel_tol));
            }
        }
    }
    Ok(())
}

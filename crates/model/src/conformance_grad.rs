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
use crate::conformance::{batch, build, build_exact, trained_rank, Subject, BATCH};
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

/// Two backward passes over the batch already set, with no zeroing between them, leave twice the gradient of one:
/// gradient accumulation (`grad_accum` micro-batches into one optimizer step) adds every micro-batch's gradient
/// instead of keeping the last. Returns the trainable tensors whose accumulated gradient is not twice the single one.
pub fn backwards_accumulate<M: Model>(m: &M) -> Result<(), String> {
    let once = gradients(m);
    m.forward();
    m.backward();
    m.poll_wait();
    let mut wrong = Vec::new();
    for (name, g1) in &once {
        let g2 = m.read_grad(name);
        let scale = max_abs(g1).max(1e-12);
        let off = g1.iter().zip(&g2).map(|(a, b)| (2.0 * a - b).abs()).fold(0.0f32, f32::max);
        if off > 1e-3 * scale {
            wrong.push(format!("{name} (|2 g1 - g2| = {off:e}, |g1| = {scale:e})"));
        }
    }
    if wrong.is_empty() {
        Ok(())
    } else {
        Err(format!("a second backward does not add its gradient: {}", wrong.join(", ")))
    }
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

/// C3b: extra targets per position ([`crate::training::TokenObjectives::enable_sparse_targets`]) on a model that
/// declares them, or a refusal on one that does not. An extra target `(v, c)` adds `c * (-log p(v))` to the loss, which
/// is exactly what the label `v` with weight `c` adds, so with three extra targets per supervised position (a random
/// id, the same id again, the label itself; signed coefficients) the model's one backward must equal the sum of four
/// weighted-loss backwards: the labels at their weights, and each slot's ids as labels at its coefficients, all over
/// one denominator. The read-back log-probability at a slot is the one a batch with that slot's ids as labels reads.
/// At the unit logit scale and at `1/2`. These are identities between two paths of the model, exact up to rounding,
/// so they hold however small a tensor's gradient is at a random initialisation; the model is built at the exact fp32
/// arithmetic ([`build_exact`]) on every device.
pub fn sparse_targets_match_their_label_equivalents<M: Model>(subject: &Subject<M>, seed: u64) -> Result<(), String> {
    const K: usize = 3;
    let (mut m, _) = build_exact(subject, trained_rank(subject), BATCH, seed);
    randomize_adapters(&m, seed);
    let vocab = subject_vocab(subject);
    let (tokens, targets) = batch(vocab);
    let objectives = require_token_objectives(&mut m).map_err(|e| e.to_string())?;
    objectives.enable_weighted_loss();
    match (objectives.enable_sparse_targets(K as u32), subject.declares.sparse_targets) {
        (Ok(()), true) => {}
        (Err(_), false) => return Ok(()),
        (Ok(()), false) => return Err(format!("{}: declares no extra targets yet enable_sparse_targets succeeded", subject.name)),
        (Err(e), true) => return Err(format!("{}: declares extra targets but refused them: {e}", subject.name)),
    }
    if objectives.enable_sparse_targets(K as u32 + 1).is_ok() {
        return Err(format!("{}: enable_sparse_targets accepted a second k", subject.name));
    }
    let n = tokens.len();
    let weights: Vec<f32> = (0..n).map(|i| if i % 3 == 0 { 0.0 } else { 0.5 + 0.1 * i as f32 }).collect();
    let mut rng = Rng::new(seed ^ 0x51ab);
    let supervised = |i: usize| targets[i] != crate::IGNORE;
    let other: Vec<u32> = (0..n).map(|i| if supervised(i) { (targets[i] + 1 + (rng.next_f32() * (vocab - 1) as f32) as u32 % (vocab - 1)) % vocab } else { crate::IGNORE }).collect();
    // Slot j's ids as a label row: the random id twice, then the label.
    let slot_ids: [Vec<u32>; K] = [other.clone(), other.clone(), targets.clone()];
    let slot_coefs: [Vec<f32>; K] = std::array::from_fn(|_| (0..n).map(|i| if supervised(i) { 2.0 * rng.next_f32() - 1.0 } else { 0.0 }).collect());
    let ids: Vec<u32> = (0..n).flat_map(|i| slot_ids.iter().map(move |s| s[i])).collect();
    let coefs: Vec<f32> = (0..n).flat_map(|i| slot_coefs.iter().map(move |c| c[i])).collect();
    let denominator = 7.0f32;
    let weighted = |m: &M, labels: &[u32], w: &[f32]| {
        m.set_batch(Batch::LmWeighted { tokens: &tokens, targets: labels, weights: w });
        m.token_objectives().expect("checked above").set_loss_denominator(denominator);
    };
    for scale in [1.0f32, 0.5] {
        require_token_objectives(&mut m).map_err(|e| e.to_string())?.set_logit_scale(scale);
        // The label-equivalent sum, and each slot's log-probabilities read as labels.
        weighted(&m, &targets, &weights);
        let mut want = gradients(&m);
        let mut want_lp = Vec::with_capacity(K);
        for (labels, c) in slot_ids.iter().zip(&slot_coefs) {
            weighted(&m, labels, c);
            // Read right after a forward: a model may drop its forward state in the backward.
            m.forward();
            want_lp.push(m.token_objectives().expect("checked above").batch_token_logprobs());
            for ((_, total), (_, g)) in want.iter_mut().zip(gradients(&m)) {
                total.iter_mut().zip(&g).for_each(|(t, g)| *t += g);
            }
        }
        weighted(&m, &targets, &weights);
        let o = m.token_objectives().expect("checked above");
        o.set_sparse_target_ids(&ids)?;
        o.set_sparse_target_weights(&coefs)?;
        m.forward();
        let lps = m.token_objectives().expect("checked above").batch_sparse_target_logprobs().ok_or_else(|| format!("{}: no extra-target log-probabilities after a forward", subject.name))?;
        let got = gradients(&m);
        for i in (0..n).filter(|&i| supervised(i)) {
            for (j, lp) in want_lp.iter().enumerate() {
                if (lps[i * K + j] - lp[i]).abs() > 1e-5 * lp[i].abs().max(1.0) {
                    return Err(format!("{}: position {i} slot {j} at logit scale {scale}: extra-target log-probability {} vs {} as a label", subject.name, lps[i * K + j], lp[i]));
                }
            }
        }
        let scale_all = want.iter().map(|(_, w)| max_abs(w)).fold(0.0, f32::max).max(1e-12);
        for ((name, g), (_, w)) in got.iter().zip(&want) {
            let off = g.iter().zip(w).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            if off > 1e-4 * scale_all {
                return Err(format!("{}: {name} at logit scale {scale}: the gradient with extra targets is {off:e} from the sum of its label equivalents (largest gradient {scale_all:e})", subject.name));
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

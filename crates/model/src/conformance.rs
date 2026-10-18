// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Conformance suite for the training capabilities a [`Model`] declares.
//!
//! A model registers one [`Subject`] (a tiny config with distinct non-square
//! dimensions, plus what it declares to provide) and runs [`run`]. The suite
//! then checks the capability contracts, not any one architecture:
//!
//! * C1 per-token log-probabilities equal the log-softmax of `logits_all` at
//!   the supervised positions, with and without a logit scale;
//! * C2 an all-ones [`Batch::LmWeighted`] reproduces plain CE loss and gradients;
//! * C3 the weighted loss's gradient matches finite differences on the
//!   trainable parameters (rank-2 LoRA factors for an adapter model);
//! * C4 zero-initialised LoRA `B` leaves the logits bit-identical to the base;
//! * C6 save, reload, identical log-probabilities;
//! * C9 every declared frozen-base tier gives gradients within its stated
//!   tolerance of the f32 base.
//!
//! The objective-level checks (C7, one SFT/DPO/GRPO/distill step each) need the
//! objectives themselves and live in `rl::conformance`; C8 (chunked head CE
//! equals unchunked) is registered by `rl/tests/conformance.rs`.
//!
//! A capability a model declares absent is asserted absent ([`Declares`]): the
//! suite fails if the model offers what it says it does not, and fails if it
//! says it offers what it does not, so a skip is always a statement the test
//! has checked.
//!
//! Swedish Embedded AB implements solutions for verifying that every model in
//! a training stack honours the same fine-tuning contract, for its clients. If
//! your team needs expertise in conformance testing for ML training code you
//! can procure our services by sending an email to info@swedishembedded.com.

use std::collections::HashMap;

use crate::logprobs::row_logprob;
use crate::training::{require_token_objectives, Capability, Unsupported};
use crate::{Batch, Model, IGNORE};

/// Rows and positions of the conformance batch.
pub const BATCH: u32 = 2;
pub const SEQ: u32 = 6;

/// Which capabilities a model claims. Every claim is checked in both
/// directions.
#[derive(Clone, Copy, Debug)]
pub struct Declares {
    /// [`Model::token_objectives`] returns `Some`.
    pub token_objectives: bool,
    /// The config accepts an adapter rank and the model trains only the
    /// `.lora_a`/`.lora_b` tensors when it is given (full fine-tune otherwise).
    pub adapters: bool,
}

/// One model under test.
pub struct Subject<M: Model> {
    pub name: &'static str,
    /// A tiny config with distinct, non-square dimensions. `Some(rank)` puts a
    /// rank-`rank` LoRA on every adapter site; `None` is the plain model.
    /// Called with `Some` only when [`Declares::adapters`].
    pub config: fn(adapter_rank: Option<u32>) -> M::Config,
    pub declares: Declares,
    /// Rebuild a model from a file written by [`Model::save`], sized for
    /// `(rows, seq)`.
    pub load: fn(path: &str, rows: u32, seq: u32) -> M,
    /// The frozen-base tiers (besides the f32 reference) the model offers
    /// for training; each is checked by C9.
    pub tiers: Vec<Tier<M>>,
}

/// A model's initial weights: master values by parameter name.
pub type InitWeights = HashMap<String, Vec<f32>>;

/// A frozen-base storage tier: how to build the model with its base held at
/// that tier, and how far its gradients may stray from the f32 base's.
pub struct Tier<M: Model> {
    pub name: &'static str,
    /// Build from the same config and weights as the f32 reference, for
    /// `(rows, seq)`, with the adapter rank the suite uses.
    pub build: fn(M::Config, rows: u32, seq: u32, init: &InitWeights) -> M,
    /// Largest allowed `||g_tier - g_f32|| / ||g_f32||` over each trainable tensor.
    pub grad_rel_tol: f32,
}

/// The LoRA rank the suite builds adapter models with.
pub const RANK: u32 = 2;

/// Tokens and next-token targets of the conformance batch: `BATCH` rows of
/// `SEQ` positions; the last position of each row and one more position are
/// masked, so masking is exercised.
pub fn batch(vocab: u32) -> (Vec<u32>, Vec<u32>) {
    let n = (BATCH * SEQ) as usize;
    let tokens: Vec<u32> = (0..n as u32).map(|i| (i * 5 + 1) % vocab).collect();
    let mut targets: Vec<u32> = (0..n as u32).map(|i| (i * 5 + 6) % vocab).collect();
    for row in 0..BATCH as usize {
        targets[row * SEQ as usize + SEQ as usize - 1] = IGNORE;
    }
    targets[2] = IGNORE;
    (tokens, targets)
}

pub fn build<M: Model>(subject: &Subject<M>, rank: Option<u32>, rows: u32, seed: u64) -> (M, std::collections::HashMap<String, Vec<f32>>) {
    let cfg = (subject.config)(rank);
    let init = M::init_weights(&cfg, seed);
    (M::new(cfg, rows, SEQ, &init), init)
}

/// The rank the suite trains with: the adapter rank when the model declares
/// adapters, else the plain model.
pub(crate) fn trained_rank<M: Model>(subject: &Subject<M>) -> Option<u32> {
    subject.declares.adapters.then_some(RANK)
}

/// Run every check that needs only [`Model`]. `seed` draws the initial weights;
/// the caller reports it.
pub fn run<M: Model>(subject: &Subject<M>, seed: u64) -> Result<(), String> {
    check_declaration(subject, seed)?;
    if subject.declares.token_objectives {
        logprobs_follow_logit_scale(subject, seed)?;
        crate::conformance_grad::unit_weights_reproduce_plain_training(subject, seed)?;
        crate::conformance_grad::weighted_loss_gradient_matches_finite_differences(subject, seed)?;
    }
    if subject.declares.adapters {
        crate::conformance_adapter::zero_b_leaves_logits_unchanged(subject, seed)?;
    }
    crate::conformance_adapter::save_reload_keeps_logprobs(subject, seed)?;
    crate::conformance_grad::tiers_track_the_f32_gradients(subject, seed)?;
    Ok(())
}

/// The declared capabilities are exactly the offered ones.
pub fn check_declaration<M: Model>(subject: &Subject<M>, seed: u64) -> Result<(), String> {
    let (mut m, _) = build(subject, None, BATCH, seed);
    let offered = m.token_objectives().is_some();
    if offered != subject.declares.token_objectives {
        return Err(format!("{}: declares token_objectives={} but the model offers {}", subject.name, subject.declares.token_objectives, offered));
    }
    if !subject.declares.token_objectives {
        match require_token_objectives(&mut m) {
            Err(Unsupported { capability: Capability::TokenObjectives, .. }) => {}
            Err(other) => return Err(format!("{}: refused with the wrong capability: {other}", subject.name)),
            Ok(_) => return Err(format!("{}: declares no token objectives yet require_token_objectives succeeded", subject.name)),
        }
    }
    Ok(())
}

/// C1: `batch_token_logprobs` is `log_softmax(scale * logits)` at the supervised
/// positions and `0` at masked ones, for the unit scale and for `1/T`.
pub fn logprobs_follow_logit_scale<M: Model>(subject: &Subject<M>, seed: u64) -> Result<(), String> {
    let (mut batched, init) = build(subject, None, BATCH, seed);
    let cfg = (subject.config)(None);
    let (tokens, targets) = batch(crate::ModelConfig::vocab(&cfg));
    // Reference logits come from a single-row model on the same weights: not
    // every model's `logits_all` takes a batch.
    let single = M::new(cfg, 1, SEQ, &init);
    let rows: Vec<Vec<f32>> = (0..BATCH as usize)
        .map(|r| {
            let row = &tokens[r * SEQ as usize..(r + 1) * SEQ as usize];
            single.logits_all(row).ok_or_else(|| format!("{}: logits_all is None on a token-classification model", subject.name))
        })
        .collect::<Result<_, _>>()?;
    let vocab = rows[0].len() / SEQ as usize;

    for temperature in [1.0f32, 2.0] {
        let scale = 1.0 / temperature;
        require_token_objectives(&mut batched).map_err(|e| e.to_string())?.set_logit_scale(scale);
        batched.set_batch(Batch::Lm { tokens: &tokens, targets: &targets });
        batched.forward();
        let got = batched.token_objectives().expect("checked above").batch_token_logprobs();
        for i in 0..tokens.len() {
            let want = if targets[i] == IGNORE {
                0.0
            } else {
                let (r, t) = (i / SEQ as usize, i % SEQ as usize);
                let scaled: Vec<f32> = rows[r][t * vocab..(t + 1) * vocab].iter().map(|z| z * scale).collect();
                row_logprob(&scaled, targets[i] as usize)
            };
            if (got[i] - want).abs() > 1e-4 {
                return Err(format!("{}: position {i} at logit scale {scale}: batch_token_logprobs {} vs log_softmax {want}", subject.name, got[i]));
            }
        }
    }
    Ok(())
}

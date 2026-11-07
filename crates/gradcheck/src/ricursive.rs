// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gradient-correctness gates for decoders with learned residual mixing, gated
// attention and clipped feed-forwards for its clients. If your team needs expertise in proving that a hybrid
// local/global attention model's backward pass is right then you can procure our services by sending an email
// to info@swedishembedded.com.

//! Gradient checks for the Ripple training graph.
//!
//! What each guards, beyond the plain wiring:
//!
//! * `model.embed_tokens.weight` receives the head's path, the stream's path through every layer and the
//!   value-residual path of every layer (`x0` feeds each layer's value projection), so its gradient is a sum of
//!   contributions a backward could drop one of;
//! * `value_res_lambda` is one scalar per layer whose gradient is a reduction over every row and channel;
//! * both clamps are active on part of the feed-forward activations, so the check sees the pass-through regime
//!   and the saturated regime (zero gradient) in one batch;
//! * the sliding layers' window excludes keys (3 of up to 8), the global layer's does not.

use ricursive::{Ripple, TrainConfig};

use crate::{directional_check, elementwise_check_at, Report};

/// The batch every Ripple check differentiates: two rows of 6 tokens, the last position of each masked.
pub fn batch(vocab: u32) -> (Vec<u32>, Vec<u32>) {
    let x: Vec<u32> = (0..12).map(|i| (i * 7 + 1) % vocab).collect();
    let mut y: Vec<u32> = (0..12).map(|i| (i * 7 + 9) % vocab).collect();
    y[5] = ricursive::train::IGNORE;
    y[11] = ricursive::train::IGNORE;
    (x, y)
}

/// A tiny Ripple at trained-model weight scale, the batch set.
pub fn tiny(cfg: TrainConfig, seed: u64) -> Ripple {
    let init = cfg.tiny_weights(seed);
    let model = <Ripple as model::Model>::new(cfg, 2, 6, &init);
    let (x, y) = batch(32);
    model::Model::set_batch(&model, model::Batch::Lm { tokens: &x, targets: &y });
    model
}

/// Directional check of every parameter of the tiny full fine-tune.
pub fn check_ricursive(seed: u64) -> Report {
    let model = tiny(TrainConfig::tiny(), seed);
    directional_check(&model, 1e-3, 8, seed ^ 0x61c2)
}

/// Entry-by-entry check of the tensors a reverse pass folds: the embedding rows the batch touches (and one it does
/// not), and every layer's value-residual gate.
pub fn check_ricursive_folded(seed: u64) -> Report {
    let cfg = TrainConfig::tiny();
    let layers = cfg.model.num_hidden_layers;
    let model = tiny(cfg, seed);
    let (x, _) = batch(32);
    let hidden = 64;
    // Every channel of three touched rows and one untouched row.
    let mut rows: Vec<u32> = x.iter().copied().take(3).collect();
    rows.push((0..32).find(|v| !x.contains(v)).expect("the batch leaves a row of the table unused"));
    let indices: Vec<usize> = rows.iter().flat_map(|&r| (0..hidden).step_by(5).map(move |c| r as usize * hidden + c)).collect();
    let mut report = elementwise_check_at(&model, "model.embed_tokens.weight", 2e-3, &indices);
    for layer in 0..layers {
        let lam = elementwise_check_at(&model, &format!("model.layers.{layer}.self_attn.value_res_lambda"), 2e-3, &[0]);
        report.checks.extend(lam.checks);
    }
    report
}

/// How far the loss moves when one Ripple feature is switched off, on the same weights and batch: the proof that
/// the checks above differentiate something. Each entry is `(feature, |loss - baseline|)`.
pub fn feature_effects(seed: u64) -> Vec<(&'static str, f32)> {
    let base = TrainConfig::tiny();
    let weights = base.tiny_weights(seed);
    let (x, y) = batch(32);
    let loss = |cfg: TrainConfig| -> f32 {
        let model = <Ripple as model::Model>::new(cfg, 2, 6, &weights);
        model::Model::set_batch(&model, model::Batch::Lm { tokens: &x, targets: &y });
        model::Model::forward(&model)
    };
    let reference = loss(base.clone());
    let with = |edit: &dyn Fn(&mut ricursive::RicursiveConfig)| {
        let mut cfg = base.clone();
        edit(&mut cfg.model);
        (loss(cfg) - reference).abs()
    };
    vec![
        ("the clamp on the gate and up branches", with(&|c| c.ffn_clamp_threshold = None)),
        ("the sliding window", with(&|c| c.sliding_window = 1 << 20)),
        ("the value residual", with(&|c| c.use_valres = false)),
        ("the attention gate", with(&|c| c.use_gated_attention = false)),
        ("the embedding scale", with(&|c| c.embed_scale = None)),
    ]
}

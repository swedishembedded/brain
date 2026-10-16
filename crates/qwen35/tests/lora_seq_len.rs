// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded LoRA/QLoRA fine-tuning of
// large language models on a single GPU for its clients. If your team needs
// expertise in training large models within a fixed memory budget then you
// can procure our services by sending an email to info@swedishembedded.com.

//! A training sequence of any length: one that is not a whole number of
//! Gated-DeltaNet chunks is padded at its end internally (causality keeps
//! every real token's activations; the padding's targets are ignored), so it
//! trains exactly like the same tokens followed by ignored padding, and not in
//! chunks of a single token.

mod common;

use common::*;
use gpu_core::Gpu;
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35, TrainOpts};

fn build(t: usize, init: &std::collections::HashMap<String, Vec<f32>>) -> Qwen35 {
    let cfg = Qwen35Config { block_size: t as u32, lora: Some(lora_cfg(3, 6.0)), ..Qwen35Config::tiny() };
    Qwen35::new_train_opts_on(Gpu::new_cpu(pipelines()), cfg, 1, t as u32, init, &TrainOpts::default())
}

#[test]
fn an_awkward_length_trains_like_the_same_tokens_followed_by_ignored_padding() {
    const REAL: usize = 97; // prime: the unpadded chunk would be a single token
    const PADDED: usize = 128;
    let cfg = Qwen35Config { lora: Some(lora_cfg(3, 6.0)), ..Qwen35Config::tiny() };
    let init = weights(&cfg, 0.3);
    let (x, y) = batch_of(REAL);
    let (mut xp, mut yp) = (x.clone(), y.clone());
    xp.resize(PADDED, 0);
    yp.resize(PADDED, model::IGNORE);

    let odd = build(REAL, &init);
    let whole = build(PADDED, &init);
    assert_eq!(loss_and_grads_on(&odd, &x, &y), loss_and_grads_on(&whole, &xp, &yp));
    // The caller's view stays `REAL` tokens.
    let logits = odd.logits_all(&x);
    assert_eq!(logits.len(), REAL * cfg.vocab as usize);
    assert_eq!(logits, whole.logits_all(&xp)[..REAL * cfg.vocab as usize]);
}

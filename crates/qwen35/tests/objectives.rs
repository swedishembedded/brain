// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for reward-driven and preference
// fine-tuning of hybrid linear-attention language models on a single GPU for
// its clients. If your team needs expertise in DPO, GRPO or distillation over
// Gated-DeltaNet decoders then you can procure our services by sending an
// email to info@swedishembedded.com.

//! The per-token objectives of `qwen35::Qwen35` through a chunked LM head: a
//! weighted loss under a logit scale gives the same loss, log-probabilities
//! and gradients whatever the chunking, and its gradient is the derivative of
//! the loss it reports.

mod common;

use common::*;
use gpu_core::Gpu;
use model::training::TokenObjectives;
use qwen35::config::{lora_cfg, Qwen35Config};
use qwen35::model::{pipelines, Qwen35, TrainOpts};

const LONG: usize = 128;
const SCALE: f32 = 0.6;

fn config() -> Qwen35Config {
    Qwen35Config { block_size: LONG as u32, lora: Some(lora_cfg(3, 6.0)), ..Qwen35Config::tiny() }
}

/// A weighted-loss model with logit scale [`SCALE`] and the chunk size the budget gives.
fn build(init: &std::collections::HashMap<String, Vec<f32>>, head_budget_words: Option<u64>) -> Qwen35 {
    let opts = TrainOpts { head_budget_words, ..TrainOpts::default() };
    let mut m = Qwen35::new_train_opts_on(Gpu::new_cpu(pipelines()), config(), 1, LONG as u32, init, &opts);
    m.enable_weighted_loss();
    m.set_logit_scale(SCALE);
    m
}

/// Non-uniform per-token weights with exact zeros.
fn token_weights() -> Vec<f32> {
    (0..LONG).map(|i| ((i * 37) % 11) as f32 * 4.0).collect()
}

/// The loss, the per-token log-probabilities and the named gradients of one step.
type Step = (f32, Vec<f32>, Vec<(String, Vec<f32>)>);

fn run(m: &Qwen35, x: &[u32], y: &[u32]) -> Step {
    m.set_batch(x, y);
    m.set_loss_weights(&token_weights());
    m.zero_grads();
    let loss = m.forward();
    let logprobs = m.batch_token_logprobs();
    m.backward();
    let grads = m.param_names().into_iter().map(|n| (n.clone(), m.read_grad(&n))).collect();
    (loss, logprobs, grads)
}

/// Two chunks of 64 rows against one chunk of 128: same weighted loss, same
/// per-token log-probabilities, same adapter gradients.
#[test]
fn a_weighted_scaled_loss_is_independent_of_the_head_chunking() {
    let init = weights(&config(), 0.3);
    let (x, y) = batch_of(LONG);
    let one = run(&build(&init, None), &x, &y);
    let two = run(&build(&init, Some(64 * config().vocab as u64)), &x, &y);
    assert_eq!(one.1, two.1, "log-probabilities");
    assert!(one.1.iter().all(|&lp| lp < 0.0), "every token has a log-probability under the scaled logits");
    assert_same_training_signal((two.0, two.2), (one.0, one.2), 1e-5, 0.99999);
}

/// The reported loss is the weighted mean `sum_i w_i nll_i / count` of the
/// log-probabilities the model reports, under the scaled distribution.
#[test]
fn the_weighted_loss_is_the_weighted_mean_of_the_reported_log_probabilities() {
    let init = weights(&config(), 0.3);
    let (x, y) = batch_of(LONG);
    let (loss, logprobs, _) = run(&build(&init, Some(64 * config().vocab as u64)), &x, &y);
    let w = token_weights();
    let want: f32 = logprobs.iter().zip(&w).map(|(lp, w)| -lp * w).sum::<f32>() / y.len() as f32;
    assert!((loss - want).abs() < 1e-4 * want.abs().max(1.0), "{loss} vs {want}");
}

/// The gradient is the derivative of the loss: stepping a trained tensor
/// along its own gradient changes the loss at the rate `|g|^2`, through a
/// chunked head, a logit scale and per-token weights at once.
#[test]
fn the_gradient_is_the_derivative_of_the_weighted_scaled_loss_through_a_chunked_head() {
    let init = weights(&config(), 0.3);
    let (x, y) = batch_of(LONG);
    let m = build(&init, Some(64 * config().vocab as u64));
    let (_, _, grads) = run(&m, &x, &y);
    // The two tensors with the largest gradients: at a random initialisation
    // many adapters sit behind nearly closed gates and have no signal to difference.
    let mut by_norm: Vec<(&String, &Vec<f32>, f32)> = grads.iter().map(|(n, g)| (n, g, g.iter().map(|v| v * v).sum::<f32>())).collect();
    by_norm.sort_by(|a, b| b.2.total_cmp(&a.2));
    for &(name, g, norm2) in by_norm.iter().take(2) {
        println!("{name}: |g|^2 = {norm2}");
        assert!(norm2 > 1e-5, "{name}: the largest gradient is too small to difference ({norm2})");
        let w0 = m.read_weight(name);
        let loss_at = |sign: f32, eps: f32| {
            let moved: Vec<f32> = w0.iter().zip(g).map(|(w, g)| w + sign * eps * g).collect();
            m.write_weight(name, &moved);
            m.set_batch(&x, &y);
            m.set_loss_weights(&token_weights());
            m.forward()
        };
        // Central differences at steps h and h/2, combined to cancel the h^2 term: a single
        // step is either biased by the curvature or buried in f32 noise at this loss scale.
        let central = |h: f32| (loss_at(1.0, h) - loss_at(-1.0, h)) / (2.0 * h);
        let h = 2e-2 / norm2.sqrt();
        let numeric = (4.0 * central(h / 2.0) - central(h)) / 3.0;
        m.write_weight(name, &w0);
        // Wrong by a factor (a missing logit scale of 0.6, a missing weight) is off by 40% or
        // more; the bound leaves room for the f32 noise of a loss this size.
        assert!((numeric - norm2).abs() < 0.15 * norm2, "{name}: finite difference {numeric} vs |g|^2 {norm2}");
    }
}

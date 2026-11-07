// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A two-stage `model::Pipeline<Ripple>` trains exactly as the one-stage model does.
//!
//! Swedish Embedded AB implements solutions for training very large dense decoders across the GPUs a team
//! actually has, with the same result as one device. If your team needs expertise in pipeline-parallel training
//! of hybrid-attention language models then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! The cut falls between the second sliding layer and the global layer, so the value residual (every layer
//! reads the embedding) reaches across it: the second stage recomputes `x0` from the tokens and its share of the
//! embedding's gradient is summed with the first stage's. The loss, every parameter's gradient and a few AdamW
//! steps must equal the single stage's, for a full fine-tune, for an adapter, and with the residual stream held
//! in bf16.

use model::adapter::device::LoraCfg;
use model::saved::{SavedFormat, SavedPolicy};
use model::{Adam, Batch, Model, Pipeline, Shard};
use ricursive::train::default_targets;
use ricursive::{Ripple, TrainConfig};

const ROWS: u32 = 2;
const SEQ: u32 = 6;

fn batch(vocab: u32) -> (Vec<u32>, Vec<u32>) {
    let n = (ROWS * SEQ) as usize;
    let x: Vec<u32> = (0..n as u32).map(|i| (i * 7 + 1) % vocab).collect();
    let mut y: Vec<u32> = (0..n as u32).map(|i| (i * 7 + 9) % vocab).collect();
    y[5] = ricursive::train::IGNORE;
    (x, y)
}

fn cut() -> Vec<Shard> {
    vec![Shard { start: 0, end: 2, embed: true, head: false, gpu_index: Shard::ANY_GPU }, Shard { start: 2, end: 3, embed: false, head: true, gpu_index: Shard::ANY_GPU }]
}

fn scaled(a: &[f32], b: &[f32]) -> f32 {
    let top = b.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / top
}

/// Loss and every trained tensor's gradient of the one-stage and the two-stage model on one batch, the worst
/// scaled gradient error and the loss error.
fn compare(cfg: &TrainConfig, saved: Option<SavedPolicy>) -> (f32, f32) {
    let mut weights = cfg.tiny_weights(3);
    // An adapter starts as a no-op (B = 0), which would leave A without a gradient to compare.
    let mut rng = data::rng::Rng::new(5); // seed: both builds read the same weights
    for w in weights.iter_mut().filter(|(n, _)| n.ends_with(".lora_b")).map(|(_, w)| w) {
        w.iter_mut().for_each(|x| *x = 0.3 * rng.next_gaussian() as f32);
    }
    let (x, y) = batch(cfg.model.vocab_size as u32);
    let mut single = Ripple::new(cfg.clone(), ROWS, SEQ, &weights);
    let mut pipe = Pipeline::<Ripple>::with_shards(cfg.clone(), ROWS, SEQ, &weights, cut());
    if let Some(policy) = saved {
        single.set_saved_policy(policy).expect("one stage narrows");
        for i in 0..pipe.n_stages() {
            pipe.stage_mut(i).set_saved_policy(policy).expect("a stage narrows");
        }
    }
    single.set_batch(Batch::Lm { tokens: &x, targets: &y });
    single.zero_grads();
    let want = single.forward();
    single.backward();
    single.poll_wait();
    pipe.zero_grads();
    let got = pipe.forward(Batch::Lm { tokens: &x, targets: &y });
    pipe.backward();

    let (mut worst, mut name) = (0f32, String::new());
    for n in single.param_names() {
        let e = scaled(&pipe.reduced_grad(&n), &single.read_grad(&n));
        if e > worst {
            (worst, name) = (e, n);
        }
    }
    let loss_err = (got - want).abs() / want.abs();
    eprintln!("loss {want} vs {got} ({loss_err:.2e}); worst gradient error {worst:.2e} ({name})");
    (loss_err, worst)
}

/// Spec: a full fine-tune's loss and every gradient (the embedding's included, which both stages contribute to)
/// equal the one-stage model's.
#[test]
fn a_two_stage_full_fine_tune_has_the_one_stage_loss_and_gradients() {
    let (loss, grad) = compare(&TrainConfig::tiny(), None);
    assert!(loss < 1e-6 && grad < 1e-5, "loss {loss:.2e}, gradient {grad:.2e}");
}

/// Spec: an adapter's factors on both sides of the cut get the one-stage gradients.
#[test]
fn a_two_stage_adapter_has_the_one_stage_loss_and_gradients() {
    let (loss, grad) = compare(&TrainConfig::tiny().with_lora(LoraCfg::new(2, 4.0, default_targets())), None);
    assert!(loss < 1e-6 && grad < 1e-5, "loss {loss:.2e}, gradient {grad:.2e}");
}

/// Spec: with the stream held in bf16 between layers, the stage input is stored as the stage that wrote it
/// stored it, so the cut changes nothing.
#[test]
fn a_two_stage_run_with_bf16_boundaries_has_the_one_stage_result() {
    let (loss, grad) = compare(&TrainConfig::tiny(), Some(SavedPolicy::recompute_with(SavedFormat::Bf16)));
    assert!(loss < 1e-6 && grad < 1e-5, "loss {loss:.2e}, gradient {grad:.2e}");
}

/// Spec: AdamW over the pipeline (the global-norm clip spans both stages, the embedding's two copies take one
/// update) follows the one-stage trajectory.
#[test]
fn a_two_stage_adamw_trajectory_follows_the_one_stage_run() {
    let cfg = TrainConfig::tiny();
    let weights = cfg.tiny_weights(3);
    let (x, y) = batch(cfg.model.vocab_size as u32);
    let single = Ripple::new(cfg.clone(), ROWS, SEQ, &weights);
    let mut pipe = Pipeline::<Ripple>::with_shards(cfg, ROWS, SEQ, &weights, cut());
    let mut worst = 0f32;
    for step in 1..=4 {
        single.set_batch(Batch::Lm { tokens: &x, targets: &y });
        single.zero_grads();
        let want = single.forward();
        single.backward();
        single.adamw_step(step, 1e-2, 0.01, Adam::default(), Some(1.0), 1.0);
        single.poll_wait();
        pipe.zero_grads();
        let got = pipe.forward(Batch::Lm { tokens: &x, targets: &y });
        pipe.backward();
        pipe.adamw_step(step, 1e-2, 0.01, Adam::default(), Some(1.0), 1.0);
        worst = worst.max((got - want).abs() / want.abs());
    }
    eprintln!("worst loss difference over four steps {worst:.2e}");
    assert!(worst < 1e-5, "{worst:.2e}");
}

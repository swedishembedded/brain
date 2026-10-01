// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning of vision-language models for
// its clients. If your team needs expertise in adapting multimodal models
// to your images and conversations then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Fine-tuning the composite: a LoRA decoder and a trainable aligner over
//! frozen tower features. On a tiny random model (no checkpoint): a fresh
//! adapter changes nothing, the aligner's gradient is the loss's gradient
//! (finite differences through the whole split aligner, the splice and the
//! decoder), and a few steps overfit an image-and-reply pair while moving the
//! aligner as well as the adapters.

use std::collections::HashMap;

use data::rng::Rng;
use deepseekvl::train::{Hyper, Prepared, Trainer};
use model::projector::ProjectorConfig;
use qwen3::{Dtype, LoraCfg, QwenConfig, IGNORE};

fn gpu_disabled() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

const ROWS: usize = 3;
const BLOCK: u32 = 12;

fn decoder(lora: bool) -> QwenConfig {
    QwenConfig { block_size: BLOCK, lora: lora.then(|| LoraCfg { rank: 4, alpha: 8.0, targets: ["wq", "wk", "wv", "wo", "gate", "up", "down"].iter().map(|s| s.to_string()).collect() }), ..QwenConfig::tiny() }
}

/// The hybrid split aligner over two 8-wide streams into the decoder's 16.
fn aligner() -> (ProjectorConfig, HashMap<String, Vec<f32>>) {
    let cfg = ProjectorConfig::from_type("low_high_hybrid_split_mlp_gelu", 2, 8, 16).unwrap();
    let mut rng = Rng::new(9);
    let w = cfg.param_list().into_iter().map(|(n, len)| (n, (0..len).map(|_| (rng.next_f32() - 0.5) * 0.6).collect())).collect();
    (cfg, w)
}

/// One image (`ROWS` rows, at residual rows 1..4) and a reply the image
/// implies: BOS, the image, four reply tokens.
fn example() -> Prepared {
    let mut rng = Rng::new(4);
    let mut image = || -> Vec<Vec<f32>> { (0..2).map(|_| (0..ROWS * 8).map(|_| rng.next_f32() - 0.5).collect()).collect() };
    Prepared { images: vec![image()], tokens: vec![0, 5, 5, 5, 7, 11, 17, 3], targets: vec![IGNORE, IGNORE, IGNORE, 7, 11, 17, 3, 9], row0s: vec![1] }
}

/// Two images (`ROWS` rows each, at residual rows 1..4 and 6..9) and a reply
/// that depends on both.
fn two_images() -> Prepared {
    let mut rng = Rng::new(6);
    let mut image = || -> Vec<Vec<f32>> { (0..2).map(|_| (0..ROWS * 8).map(|_| rng.next_f32() - 0.5).collect()).collect() };
    let images = vec![image(), image()];
    Prepared { images, tokens: vec![0, 5, 5, 5, 7, 8, 5, 5, 5, 11, 17, 3], targets: vec![IGNORE, IGNORE, IGNORE, IGNORE, IGNORE, IGNORE, IGNORE, IGNORE, 11, 17, 3, 9], row0s: vec![1, 6] }
}

fn trainer(lora: bool, seed: u64) -> Trainer {
    let cfg = decoder(lora);
    let base = qwen3::init_weights(&decoder(false), 11);
    let (acfg, aw) = aligner();
    Trainer::new(cfg, Box::new(base), Dtype::F32, BLOCK, acfg, aw, ROWS, seed).unwrap()
}

#[test]
fn a_fresh_adapter_changes_nothing() {
    if gpu_disabled() {
        return;
    }
    let p = example();
    let (mut plain, mut adapted) = (trainer(false, 1), trainer(true, 1));
    let (a, b) = (plain.loss(&p), adapted.loss(&p));
    assert_eq!(a, b, "zero-delta adapters leave the loss as it was");
}

#[test]
fn the_aligner_gradient_is_the_loss_gradient() {
    if gpu_disabled() {
        return;
    }
    check_aligner_gradient(&example());
}

/// With two images the aligner serves both: its gradient sums the rows of
/// each, and still equals the loss's finite difference.
#[test]
fn the_aligner_gradient_covers_every_image_of_an_example() {
    if gpu_disabled() {
        return;
    }
    check_aligner_gradient(&two_images());
}

fn check_aligner_gradient(p: &Prepared) {
    let mut t = trainer(true, 1);
    let (_, grads) = t.loss_and_aligner_grads(p);
    let base = t.aligner_weights();
    let eps = 1e-2f32;
    let mut checked = 0;
    for (name, g) in &grads {
        let w = &base[name];
        for idx in (0..w.len()).step_by((w.len() / 4).max(1)) {
            let mut probe = |delta: f32| {
                let mut shifted = base.clone();
                shifted.get_mut(name).unwrap()[idx] += delta;
                t.set_aligner_weights(&shifted);
                t.loss(p)
            };
            let numeric = (probe(eps) - probe(-eps)) / (2.0 * eps);
            let scale = g.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-3);
            assert!((numeric - g[idx]).abs() < 3e-2 * scale, "{name}[{idx}]: analytic {} vs numeric {numeric}", g[idx]);
            checked += 1;
        }
    }
    assert!(checked >= 12, "every aligner tensor was probed ({checked})");
}

#[test]
fn a_few_steps_overfit_the_pair_and_move_the_aligner() {
    if gpu_disabled() {
        return;
    }
    let p = example();
    // The whole decoder trains here: a rank-4 adapter over a random base this
    // small cannot fit five positions, which is a limit of the fixture.
    let mut t = trainer(false, 1);
    let before = t.aligner_weights();
    let first = t.loss(&p);
    let hyper = Hyper { lr: 3e-3, aligner_lr: 3e-3, weight_decay: 0.0, grad_clip: 0.0 };
    for step in 1..=150 {
        t.step(&p, step, &hyper);
    }
    let last = t.loss(&p);
    assert!(last < first * 0.1, "the pair was learned: {first} -> {last}");
    let moved: f32 = t.aligner_weights().iter().map(|(n, w)| w.iter().zip(&before[n]).map(|(a, b)| (a - b).abs()).sum::<f32>()).sum();
    assert!(moved > 1e-2, "the aligner trained too ({moved})");
}

#[test]
fn a_step_moves_the_adapters_and_lowers_the_loss() {
    if gpu_disabled() {
        return;
    }
    let p = example();
    let mut t = trainer(true, 1);
    let adapter = t.decoder().param_names().into_iter().find(|n| n.ends_with(".lora_b")).expect("an adapter tensor");
    assert!(t.decoder().read_weight(&adapter).iter().all(|x| *x == 0.0), "B starts at zero");
    let hyper = Hyper { lr: 5e-2, aligner_lr: 1e-2, weight_decay: 0.0, grad_clip: 1.0 };
    let first = t.step(&p, 1, &hyper);
    assert!(t.decoder().read_weight(&adapter).iter().any(|x| *x != 0.0), "the step trained the adapter");
    let second = t.step(&p, 2, &hyper);
    assert!(second < first, "{first} -> {second}");
}

/// A decoder cut into two pipeline stages trains the aligner exactly as the
/// whole one does: the images are spliced into the first stage and their
/// gradient read back from it, across both images of an example.
#[test]
fn a_two_stage_decoder_gives_the_aligner_the_same_loss_and_gradient() {
    if gpu_disabled() {
        return;
    }
    let p = two_images();
    let (loss, want) = trainer(true, 1).loss_and_aligner_grads(&p);

    let cfg = decoder(true);
    let base = qwen3::init_weights(&decoder(false), 11);
    let init = qwen3::finetune::LoraInit::fresh(&cfg, Box::new(base), 1);
    let shards = model::plan_balanced(&<qwen3::Qwen as model::Shardable>::shard_cost(&cfg, 1, BLOCK), &[model::Shard::ANY_GPU, model::Shard::ANY_GPU]);
    assert_eq!(shards.len(), 2);
    let pipe = model::Pipeline::<qwen3::Qwen>::with_shards_dt(cfg.clone(), 1, BLOCK, &init, shards, Dtype::F32);
    let staged = qwen3::finetune::Trained::Pipeline(model::PipelineModel::new(pipe, cfg));
    let (acfg, aw) = aligner();
    let mut t = Trainer::with_decoder(staged, BLOCK, acfg, aw, ROWS).unwrap();
    let (got_loss, got) = t.loss_and_aligner_grads(&p);

    assert!((got_loss - loss).abs() < 1e-4 * loss.abs().max(1.0), "loss {got_loss} vs {loss}");
    for (name, w) in &want {
        let scale = w.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6);
        let worst = got[name].iter().zip(w).fold(0.0f32, |m, (a, b)| m.max((a - b).abs() / scale));
        assert!(worst < 2e-3, "{name}: the two-stage gradient differs by {worst}");
    }
}

/// A step on a batch of examples is the step on their mean gradient: a batch
/// of one example twice is that example's step.
#[test]
fn a_batch_of_the_same_example_twice_steps_as_the_example_once() {
    if gpu_disabled() {
        return;
    }
    let p = two_images();
    let hyper = Hyper { lr: 5e-2, aligner_lr: 1e-2, weight_decay: 0.0, grad_clip: 1.0 };
    let (mut once, mut twice) = (trainer(true, 1), trainer(true, 1));
    let a = once.step(&p, 1, &hyper);
    let b = twice.step_batch(&[&p, &p], 1, &hyper);
    assert!((a - b).abs() < 1e-5, "the mean loss of identical examples is the example's: {a} vs {b}");
    let (wa, wb) = (once.aligner_weights(), twice.aligner_weights());
    for (name, w) in &wa {
        let worst = w.iter().zip(&wb[name]).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(worst < 1e-5, "{name}: the aligner moved differently ({worst})");
    }
    let adapter = once.decoder().param_names().into_iter().find(|n| n.ends_with(".lora_b")).unwrap();
    let worst = once.decoder().read_weight(&adapter).iter().zip(&twice.decoder().read_weight(&adapter)).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
    assert!(worst < 1e-5, "the adapter moved differently ({worst})");
}

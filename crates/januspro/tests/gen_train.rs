// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning of generative multimodal models
// for its clients. If your team needs expertise in teaching a model to draw
// your domain then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Training Janus-Pro to generate: a LoRA decoder reads a prompt, then the
//! image's own VQ tokens one at a time (each through the generation aligner
//! and the code embedding), and its hidden states go through the generation
//! head to a cross-entropy against the next token. On a tiny random model
//! (no checkpoint): a fresh adapter changes nothing, the gradient of every
//! trainable generation part (head, aligner, code embedding) equals finite
//! differences through the decoder and the splice, and the pair is overfit.

use std::collections::HashMap;

use data::rng::Rng;
use deepseekvl::train::Hyper;
use januspro::train::{GenParts, GenPrepared, GenTrainer};
use model::projector::ProjectorConfig;
use qwen3::{Dtype, LoraCfg, QwenConfig};

fn gpu_disabled() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

const BLOCK: u32 = 12;
/// Image codes per image, and the codebook.
const POSITIONS: usize = 6;
const CODES: usize = 12;
const CODE_DIM: usize = 4;

fn decoder(lora: bool) -> QwenConfig {
    QwenConfig { block_size: BLOCK, lora: lora.then(|| LoraCfg { rank: 4, alpha: 8.0, targets: ["wq", "wk", "wv", "wo", "gate", "up", "down"].iter().map(|s| s.to_string()).collect() }), ..QwenConfig::tiny() }
}

fn random(cfg: &ProjectorConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = Rng::new(seed);
    cfg.param_list().into_iter().map(|(n, len)| (n, (0..len).map(|_| (rng.next_f32() - 0.5) * 0.6).collect())).collect()
}

fn parts() -> GenParts {
    let d = QwenConfig::tiny().d_model;
    let aligner = ProjectorConfig::from_type("mlp_gelu", 2, CODE_DIM as u32, d).unwrap();
    let head = ProjectorConfig::from_type("mlp_gelu", 2, d, 12).unwrap().with_out_dim(CODES as u32).unwrap();
    let mut rng = Rng::new(5);
    GenParts {
        aligner: (aligner, random(&aligner, 6)),
        head: (head, random(&head, 7)),
        embed: (0..CODES * CODE_DIM).map(|_| rng.next_f32() - 0.5).collect(),
        code_dim: CODE_DIM,
        positions: POSITIONS,
    }
}

/// A prompt of four tokens ending in the begin-of-image tag, then the image's
/// six codes, the first five fed back as rows.
fn example() -> GenPrepared {
    let codes = vec![3u32, 7, 7, 1, 9, 4];
    let mut tokens = vec![0u32, 5, 9, 2]; // bos, the prompt, <begin_of_image>
    tokens.extend(std::iter::repeat(5).take(POSITIONS - 1));
    GenPrepared { tokens, codes }
}

fn trainer(lora: bool) -> GenTrainer {
    let base = qwen3::init_weights(&decoder(false), 11);
    GenTrainer::new(decoder(lora), Box::new(base), Dtype::F32, BLOCK, parts(), 1).unwrap()
}

#[test]
fn a_fresh_adapter_changes_nothing() {
    if gpu_disabled() {
        return;
    }
    let p = example();
    let (mut plain, mut adapted) = (trainer(false), trainer(true));
    assert_eq!(plain.loss(&p), adapted.loss(&p));
}

#[test]
fn every_trainable_generation_part_has_the_loss_gradient() {
    if gpu_disabled() {
        return;
    }
    let p = example();
    let mut t = trainer(true);
    let (_, grads) = t.loss_and_grads(&p);
    let eps = 1e-2f32;
    let mut checked = 0;
    let shift = |t: &mut GenTrainer, part: &str, name: &str, idx: usize, delta: f32| match part {
        "head" => {
            let mut w = t.head_weights();
            w.get_mut(name).unwrap()[idx] += delta;
            t.set_head_weights(&w);
        }
        "aligner" => {
            let mut w = t.aligner_weights();
            w.get_mut(name).unwrap()[idx] += delta;
            t.set_aligner_weights(&w);
        }
        _ => {
            let mut e = t.embed().to_vec();
            e[idx] += delta;
            t.set_embed(&e);
        }
    };
    let probe = |t: &mut GenTrainer, part: &str, name: &str, idx: usize, analytic: f32, scale: f32| {
        let mut numeric = [0.0f32; 2];
        for (k, delta) in [eps, -eps].into_iter().enumerate() {
            shift(t, part, name, idx, delta);
            numeric[k] = t.loss(&p);
            shift(t, part, name, idx, -delta);
        }
        let numeric = (numeric[0] - numeric[1]) / (2.0 * eps);
        assert!((numeric - analytic).abs() < 3e-2 * scale, "{part}.{name}[{idx}]: analytic {analytic} vs numeric {numeric}");
    };
    for (part, tensors) in [("head", &grads.head), ("aligner", &grads.aligner)] {
        for (name, g) in tensors {
            let scale = g.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-3);
            for idx in (0..g.len()).step_by((g.len() / 4).max(1)) {
                probe(&mut t, part, name, idx, g[idx], scale);
                checked += 1;
            }
        }
    }
    // The embedding rows of the codes that were fed back carry gradient; the
    // rest of the table, and the last code (never fed), none.
    let scale = grads.embed.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-3);
    for &code in &[3usize, 7, 1, 9] {
        probe(&mut t, "embed", "", code * CODE_DIM, grads.embed[code * CODE_DIM], scale);
        checked += 1;
    }
    for code in [0usize, 2, 4, 5, 6, 8, 10, 11] {
        assert!(grads.embed[code * CODE_DIM..(code + 1) * CODE_DIM].iter().all(|g| *g == 0.0), "code {code} was not fed back");
    }
    assert!(checked >= 20, "every part was probed ({checked})");
}

#[test]
fn a_few_steps_overfit_the_image_and_move_every_generation_part() {
    if gpu_disabled() {
        return;
    }
    let p = example();
    // The whole decoder trains: a rank-4 adapter over a random base this
    // small cannot fit six positions, a limit of the fixture.
    let mut t = trainer(false);
    let before = (t.head_weights(), t.aligner_weights(), t.embed().to_vec());
    let first = t.loss(&p);
    let hyper = Hyper { lr: 3e-3, aligner_lr: 5e-3, weight_decay: 0.0, grad_clip: 0.0 };
    for step in 1..=150 {
        t.step(&p, step, &hyper);
    }
    let last = t.loss(&p);
    assert!(last < first * 0.15, "the image was learned: {first} -> {last}");
    let moved = |a: &HashMap<String, Vec<f32>>, b: &HashMap<String, Vec<f32>>| a.iter().map(|(n, w)| w.iter().zip(&b[n]).map(|(x, y)| (x - y).abs()).sum::<f32>()).sum::<f32>();
    assert!(moved(&t.head_weights(), &before.0) > 1e-2, "the head trained");
    assert!(moved(&t.aligner_weights(), &before.1) > 1e-2, "the aligner trained");
    assert!(t.embed().iter().zip(&before.2).map(|(x, y)| (x - y).abs()).sum::<f32>() > 1e-2, "the code embedding trained");
}

/// The decoder reads the begin-of-image tag itself and predicts the image's
/// first code from it, as it does when drawing: a different tag token changes
/// the loss, and so does a different token just before it.
#[test]
fn the_first_code_is_predicted_from_the_begin_of_image_tag() {
    if gpu_disabled() {
        return;
    }
    let p = example();
    let tag = 3; // the prompt is four tokens; the tag is its last
    let mut t = trainer(false);
    let base = t.loss(&p);
    let mut other_tag = p.clone();
    other_tag.tokens[tag] = 7;
    assert_ne!(t.loss(&other_tag), base, "the tag token is an input of the position that predicts the first code");
    let mut earlier = p.clone();
    earlier.tokens[tag - 1] = 11;
    assert_ne!(t.loss(&earlier), base);
}

/// A decoder cut into two pipeline stages trains the generation parts exactly
/// as the whole one does: the code rows are spliced into the first stage, the
/// head's gradient enters at the last, and every generation gradient agrees.
#[test]
fn a_two_stage_decoder_gives_every_generation_part_the_same_loss_and_gradient() {
    if gpu_disabled() {
        return;
    }
    let p = example();
    let (loss, want) = trainer(true).loss_and_grads(&p);

    let cfg = decoder(true);
    let base = qwen3::init_weights(&decoder(false), 11);
    let init = qwen3::finetune::LoraInit::fresh(&cfg, Box::new(base), 1);
    let shards = model::plan_balanced(&<qwen3::Qwen as model::Shardable>::shard_cost(&cfg, 1, BLOCK), &[model::Shard::ANY_GPU, model::Shard::ANY_GPU]);
    assert_eq!(shards.len(), 2);
    let d_model = cfg.d_model as usize;
    let pipe = model::Pipeline::<qwen3::Qwen>::with_shards_dt(cfg.clone(), 1, BLOCK, &init, shards, Dtype::F32);
    let staged = qwen3::finetune::Trained::Pipeline(model::PipelineModel::new(pipe, cfg));
    let (got_loss, got) = GenTrainer::with_decoder(staged, d_model, BLOCK, parts()).unwrap().loss_and_grads(&p);

    assert!((got_loss - loss).abs() < 1e-4 * loss.abs().max(1.0), "loss {got_loss} vs {loss}");
    let close = |name: &str, a: &[f32], b: &[f32]| {
        let scale = b.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6);
        let worst = a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs() / scale));
        assert!(worst < 2e-3, "{name}: the two-stage gradient differs by {worst}");
    };
    for (name, w) in &want.head {
        close(name, &got.head[name], w);
    }
    for (name, w) in &want.aligner {
        close(name, &got.aligner[name], w);
    }
    close("embed", &got.embed, &want.embed);
}

/// A step on a batch of examples is the step on their mean gradient: a batch
/// of one example twice is that example's step.
#[test]
fn a_batch_of_the_same_example_twice_steps_as_the_example_once() {
    if gpu_disabled() {
        return;
    }
    let p = example();
    let hyper = Hyper { lr: 5e-2, aligner_lr: 1e-2, weight_decay: 0.0, grad_clip: 1.0 };
    let (mut once, mut twice) = (trainer(true), trainer(true));
    let a = once.step(&p, 1, &hyper);
    let b = twice.step_batch(&[&p, &p], 1, &hyper);
    assert!((a - b).abs() < 1e-5, "{a} vs {b}");
    let (wa, wb) = (once.head_weights(), twice.head_weights());
    for (name, w) in &wa {
        let worst = w.iter().zip(&wb[name]).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(worst < 1e-5, "{name}: the head moved differently ({worst})");
    }
    let adapter = once.decoder().param_names().into_iter().find(|n| n.ends_with(".lora_b")).unwrap();
    let worst = once.decoder().read_weight(&adapter).iter().zip(&twice.decoder().read_weight(&adapter)).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
    assert!(worst < 1e-5, "the adapter moved differently ({worst})");
}

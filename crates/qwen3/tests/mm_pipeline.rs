// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements multimodal fine-tuning on consumer GPUs
// for its clients. If your team needs expertise in training vision-language
// models that do not fit one card then you can procure our services by
// sending an email to info@swedishembedded.com.

//! A decoder split into pipeline stages takes images and an output head of the
//! caller's like the whole one does: the images are spliced into the first
//! stage and their gradient read back from it, the head's gradient enters at
//! the last stage and flows back through the others. Two stages agree with
//! one on the hidden states, the images' gradient and every adapter gradient.

use std::collections::HashMap;

use data::rng::Rng;
use model::{Pipeline, PipelineModel, Shard, Shardable};
use qwen3::finetune::Trained;
use qwen3::{Dtype, LoraCfg, Qwen, QwenConfig};

fn gpu_disabled() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

const REGIONS: [(u32, u32); 2] = [(1, 2), (5, 3)];
const TOKENS: [u32; 10] = [3, 9, 9, 4, 6, 9, 9, 9, 7, 2];

fn config() -> QwenConfig {
    QwenConfig { block_size: 10, tie_embeddings: false, lora: Some(LoraCfg::attn(2, 4.0)), ..QwenConfig::tiny() }
}

fn weights(cfg: &QwenConfig) -> HashMap<String, Vec<f32>> {
    let mut w = qwen3::init_weights(cfg, 23);
    for (n, v) in w.iter_mut().filter(|(n, _)| n.ends_with(".lora_b")) {
        v.iter_mut().enumerate().for_each(|(i, x)| *x = ((i * 5 + n.len()) % 9) as f32 * 0.03 - 0.12);
    }
    w
}

fn one_stage(cfg: &QwenConfig, init: &HashMap<String, Vec<f32>>) -> Trained {
    Trained::Single(Qwen::new_lora_dt(cfg.clone(), 1, 10, init, Dtype::F32))
}

fn two_stages(cfg: &QwenConfig, init: &HashMap<String, Vec<f32>>) -> Trained {
    let cost = <Qwen as Shardable>::shard_cost(cfg, 1, 10);
    let shards = model::plan_balanced(&cost, &[Shard::ANY_GPU, Shard::ANY_GPU]);
    assert_eq!(shards.len(), 2);
    let pipe = Pipeline::<Qwen>::with_shards_dt(cfg.clone(), 1, 10, init, shards, Dtype::F32);
    Trained::Pipeline(PipelineModel::new(pipe, cfg.clone()))
}

/// Everything one forward and backward produces, through the multimodal seams.
fn run(mut m: Trained, embeds: &[f32], d_hidden: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<(String, Vec<f32>)>) {
    m.enable_mm_splices(&REGIONS);
    m.enable_external_head();
    m.write_img_embeds(embeds);
    m.set_batch(&TOKENS, &[0; 10]);
    m.zero_grads();
    let hidden = m.forward_hidden();
    m.backward_hidden(d_hidden);
    let grads = m.param_names().into_iter().filter(|n| n.contains(".lora_")).map(|n| (n.clone(), m.read_grad(&n))).collect();
    (hidden, m.read_d_img_embeds(), grads)
}

fn worst(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6);
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs() / scale))
}

#[test]
fn two_stages_agree_with_one_through_the_image_and_head_seams() {
    if gpu_disabled() {
        return;
    }
    let cfg = config();
    let (d, init) = (cfg.d_model as usize, weights(&cfg));
    let mut rng = Rng::new(5);
    let rows: usize = REGIONS.iter().map(|r| r.1 as usize).sum();
    let embeds: Vec<f32> = (0..rows * d).map(|_| rng.next_gaussian() as f32 * 0.3).collect();
    let d_hidden: Vec<f32> = (0..10 * d).map(|_| rng.next_gaussian() as f32 * 0.1).collect();

    let (want_hidden, want_d_img, want_grads) = run(one_stage(&cfg, &init), &embeds, &d_hidden);
    let (hidden, d_img, grads) = run(two_stages(&cfg, &init), &embeds, &d_hidden);

    assert!(worst(&hidden, &want_hidden) < 1e-4, "hidden states differ by {}", worst(&hidden, &want_hidden));
    assert_eq!(d_img.len(), embeds.len());
    assert!(want_d_img.iter().any(|g| g.abs() > 1e-6), "the images receive gradient");
    assert!(worst(&d_img, &want_d_img) < 2e-3, "image gradient differs by {}", worst(&d_img, &want_d_img));
    assert_eq!(grads.len(), want_grads.len());
    let mut compared = 0;
    for ((name, g), (_, w)) in grads.iter().zip(&want_grads) {
        assert!(worst(g, w) < 2e-3, "{name}: gradient differs by {}", worst(g, w));
        compared += 1;
    }
    assert!(compared >= 8, "every adapter tensor was compared ({compared})");
}

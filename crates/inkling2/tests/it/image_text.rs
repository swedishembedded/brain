// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements image-text fine-tuning of multimodal models for its clients. If your team needs
// expertise in training a vision tower and a decoder together then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: image-text training. The tiny looped decoder with a tiny tower spliced in: adapters on both lower the loss of
//! a fixed image-text row, and the tower's adapters receive the decoder's gradient exactly (central differences of the
//! whole image-text loss, the decoder's routing pinned so a step cannot flip a top-k selection).

use std::collections::HashMap;

use data::rng::{random_seed, Lcg};
use imaging::RegionImage;
use inkling2::config::VisionConfig;
use inkling2::train::image_text::ImageTextTrainer;
use inkling2::train::{targets, tiny_config, Inkling2Train};
use inkling2::vision::{LoraTraining, VisionTower, VisionWeights};
use model::adapter::device::LoraCfg;
use model::Model;

/// Two layers of width 64 with regions of 4x4 pixels, projecting to the tiny decoder's width of 32.
fn tower_config() -> VisionConfig {
    serde_json::from_value(serde_json::json!({
        "model_type": "inkling2_vision", "text_hidden_size": 32, "temporal_patch_size": 2, "native_patch_size": 2,
        "adapter_merge_size": 2, "projection_hidden_dim": 20, "projector_bias": false, "hidden_size": 64,
        "num_hidden_layers": 2, "num_attention_heads": 4, "mlp_ratio": 2.0, "posemb_grid_size": 4, "layer_norm_eps": 1e-5
    }))
    .unwrap()
}

fn tower_weights(cfg: &VisionConfig, rng: &mut Lcg) -> VisionWeights {
    let tensors: HashMap<String, Vec<f32>> = inkling2::names::vision_manifest_of(cfg)
        .into_iter()
        .map(|e| {
            let n = e.shape.iter().product::<u64>() as usize;
            let norm = e.name.ends_with("ln_1.weight") || e.name.ends_with("ln_2.weight") || e.name.ends_with("ln_pre.weight") || e.name.ends_with("gamma");
            let v: Vec<f32> = if norm { (0..n).map(|_| 1.0 + rng.scaled(0.3)).collect() } else { rng.vec_scaled(n, 0.35) };
            (e.name, v)
        })
        .collect();
    VisionWeights::from_tensors(cfg, tensors).unwrap()
}

/// The image token of the tiny vocabulary (a valid token the text never uses).
const IMAGE: u32 = 35;
const T: usize = 8;

struct Fixture {
    trainer: ImageTextTrainer,
    tokens: Vec<u32>,
    targets: Vec<u32>,
    images: Vec<RegionImage>,
}

fn fixture(seed: u64) -> Fixture {
    let mut rng = Lcg::new(seed);
    let tc = tiny_config().with_lora(LoraCfg::new(2, 4.0, targets(false))).unwrap();
    let init = <Inkling2Train as Model>::init_weights(&tc, seed);
    let decoder = Inkling2Train::new(tc, 1, T as u32, &init);
    let vcfg = tower_config();
    let leaves: Vec<String> = ["q_proj", "k_proj", "v_proj", "out_proj", "c_fc", "c_proj", "proj_in", "proj_out"].iter().map(|s| s.to_string()).collect();
    let lc = LoraCfg::new(2, 4.0, leaves);
    let mut draw = || rng.scaled(0.3);
    let tower = VisionTower::new_trainable(decoder.gpu(), &vcfg, &tower_weights(&vcfg, &mut Lcg::new(seed ^ 1)), &|_| false, Some(LoraTraining { cfg: &lc, init: &mut draw })).unwrap();
    // One image of 3 x 1 regions: three placeholder tokens after a text token.
    let region = (vcfg.native_patch_size * vcfg.adapter_merge_size) as usize;
    let images = vec![RegionImage { rows: 3, cols: 1, pixels: Lcg::new(seed ^ 2).vec_scaled(3 * region * region * 3, 1.0) }];
    let tokens: Vec<u32> = vec![1, IMAGE, IMAGE, IMAGE, 7, 9, 11, 13];
    let targets: Vec<u32> = (0..T).map(|i| if i < 4 || i + 1 == T { model::IGNORE } else { tokens[i + 1] }).collect();
    Fixture { trainer: ImageTextTrainer::new(decoder, tower).unwrap(), tokens, targets, images }
}

#[test]
fn image_text_adapters_lower_the_loss_and_the_tower_receives_the_decoders_gradient() {
    let seed = random_seed();
    eprintln!("seed {seed}");
    let f = fixture(seed);
    let t = &f.trainer;
    let lora = t.tower().lora_store().expect("tower adapters");
    // B off zero so every factor has a gradient.
    let mut rng = Lcg::new(seed ^ 3);
    for (name, len) in lora.params.clone() {
        if name.ends_with("lora_b") {
            t.decoder().gpu().write_f32(lora.w(&name), &rng.vec_scaled(len, 0.3));
        }
    }

    // The gradient reaching the tower: central differences of the image-text loss, with the decoder's routing pinned.
    t.zero_grads();
    t.step(&f.tokens, &f.targets, &f.images, IMAGE).unwrap();
    let pinned = std::sync::Arc::new(t.decoder().taken_routing().unwrap());
    model::training::TokenObjectives::set_routing_replay(t.decoder(), Some(pinned)).unwrap();
    t.zero_grads();
    t.step(&f.tokens, &f.targets, &f.images, IMAGE).unwrap();
    let g = t.decoder().gpu();
    let mut checked = 0;
    for name in lora.params.iter().map(|(n, _)| n.clone()).filter(|n| n.contains("proj_out") || n.contains("layers.1.mlp.c_fc")) {
        let analytic = lora.read_grad(g, &name);
        let w0 = lora.read_weight(g, &name);
        let i = (0..w0.len()).max_by(|&a, &b| analytic[a].abs().total_cmp(&analytic[b].abs())).unwrap();
        assert!(analytic[i] != 0.0, "{name}: no gradient reached the tower's adapter");
        let eps = 1e-3f32;
        let loss_at = |v: f32| {
            let mut w = w0.clone();
            w[i] = v;
            g.write_f32(lora.w(&name), &w);
            t.zero_grads();
            t.step(&f.tokens, &f.targets, &f.images, IMAGE).unwrap()
        };
        let numeric = (loss_at(w0[i] + eps) - loss_at(w0[i] - eps)) / (2.0 * eps);
        g.write_f32(lora.w(&name), &w0);
        assert!((numeric - analytic[i]).abs() <= 5e-2 * analytic[i].abs().max(1e-3), "{name}[{i}]: analytic {} vs central difference {numeric}", analytic[i]);
        checked += 1;
    }
    assert!(checked >= 4, "only {checked} tower adapter tensors checked");

    // A few steps of both adapters lower the loss (the routing stays pinned: a flipped selection is another function,
    // and a loss that jumps between functions says nothing about learning).
    let mut first = None;
    let mut last = 0.0;
    for step in 1..=30 {
        t.zero_grads();
        last = t.step(&f.tokens, &f.targets, &f.images, IMAGE).unwrap();
        first.get_or_insert(last);
        t.update(step, 5e-3, 0.0, model::Adam::default(), Some(1.0));
    }
    let first = first.unwrap();
    assert!(last < 0.8 * first, "image-text loss {first} -> {last}");
}

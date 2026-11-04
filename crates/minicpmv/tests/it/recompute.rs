// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded fine-tuning of vision-language models for its clients. If your team
// needs expertise in training multimodal assistants with a vision tower within a fixed device budget, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The saved-activation policy of the MiniCPM-V trainer. The vision tower's backward always rebuilds each encoder
//! block from the block's input (gradient checkpointing), so a tower that trains accepts recomputing as what it
//! already does and refuses keeping every block and narrow saved formats by name
//! (`model::saved::check_always_recomputes`'s contract). The decoder takes the policy as `qwen35` does: with the
//! vision tower frozen below the connector, keeping and recomputing give the same loss and connector gradient, bit
//! for bit.

use data::rng::Rng;
use minicpmv::dataset::Sample;
use minicpmv::preprocess::ProcessedImage;
use minicpmv::train::{PartMode, Rates, TrainSpec, Trainer};
use minicpmv::vision::View;
use minicpmv::vision_train::train_pipelines;
use minicpmv::{Downsample, MiniCpmVConfig};
use model::saved::{SavedFormat, SavedPolicy};

fn trainer(spec: TrainSpec, seed: u64) -> Trainer {
    let cfg = MiniCpmVConfig::tiny();
    let mut rng = Rng::new(seed);
    let vision: std::collections::HashMap<String, Vec<f32>> = minicpmv::vision::tensor_manifest(&cfg)
        .into_iter()
        .map(|(n, k)| {
            let gain = n.contains("norm") && n.ends_with(".weight");
            (n, (0..k).map(|_| if gain { 1.0 } else { 0.4 * (rng.next_f32() - 0.5) }).collect())
        })
        .collect();
    let dec = qwen35::init::init_weights(&cfg.text, seed ^ 1);
    let base = qwen35::base_source::BaseCheckpoint { cfg: cfg.text.clone(), src: Box::new(dec) };
    Trainer::new(&gpu_core::testgpu::dev(train_pipelines()), cfg, &vision, base, spec).expect("trainer")
}

fn spec(decoder: PartMode, connector: bool, vision: PartMode, seed: u64) -> TrainSpec {
    TrainSpec { decoder, connector, vision, block: 12, downsample: Downsample::X16, lr: Rates::uniform(0.0), weight_decay: 0.0, grad_clip: None, seed }
}

fn sample(seed: u64) -> Sample {
    let cfg = MiniCpmVConfig::tiny();
    let mut rng = Rng::new(seed);
    let views = vec![View { h: 4, w: 4 }, View { h: 4, w: 8 }];
    let pp = 3 * cfg.vision.patch * cfg.vision.patch;
    let patches = (0..48 * pp).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let ids: Vec<u32> = vec![3, 7, 0, 9, 4, 0, 0, 11, 5, 8, 2, 6];
    let targets = (0..ids.len()).map(|i| if (8..ids.len()).contains(&(i + 1)) { ids[i + 1] } else { model::IGNORE }).collect();
    Sample { ids, targets, image_runs: vec![(2, 1), (5, 2)], video_runs: vec![], images: vec![ProcessedImage { views, patches, grid: None, resized: Vec::new() }], videos: vec![] }
}

/// The loss and the connector's gradient of one accumulating step under `policy`, as bits.
fn connector_step(seed: u64, policy: SavedPolicy) -> (u32, Vec<u32>) {
    let mut t = trainer(spec(PartMode::Lora { rank: 2, alpha: 4.0 }, true, PartMode::Frozen, seed), seed);
    t.set_saved_policy(policy).unwrap();
    let loss = t.accumulate(&sample(seed ^ 2)).unwrap();
    let grad = t.tower.read_grad("merger.fc1.weight").into_iter().map(f32::to_bits).collect();
    (loss.to_bits(), grad)
}

#[test]
fn a_training_vision_tower_always_recomputes_and_a_frozen_one_leaves_the_decoder_free() {
    let seed = data::rng::random_seed();

    // The tower trains below the connector: recomputing is what it does, nothing else is available.
    let mut t = trainer(spec(PartMode::Lora { rank: 2, alpha: 4.0 }, true, PartMode::Lora { rank: 2, alpha: 4.0 }, seed), seed);
    t.set_saved_policy(SavedPolicy::Recompute).unwrap();
    for refused in [SavedPolicy::Keep, SavedPolicy::Quantized(SavedFormat::Bf16)] {
        let err = t.set_saved_policy(refused).unwrap_err();
        assert!(err.contains("minicpmv vision tower"), "the refusal of {refused:?} does not name the tower: {err}");
    }

    // The tower is frozen: the decoder alone decides, and its two answers agree bit for bit.
    let (want_loss, want_grad) = connector_step(seed, SavedPolicy::Keep);
    let (got_loss, got_grad) = connector_step(seed, SavedPolicy::Recompute);
    assert_eq!(got_loss, want_loss, "the loss differs recomputing");
    assert!(want_grad.iter().any(|b| f32::from_bits(*b) != 0.0), "the connector received no gradient");
    assert!(got_grad == want_grad, "the connector gradient differs recomputing");
}
